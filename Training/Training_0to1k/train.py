import os
import json
import torch
import torch.nn as nn
import random
import matplotlib
# Ensure non-interactive backend to avoid any GUI/blocking issues on servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import warnings
from safetensors.torch import save_file
import time
from datetime import datetime
# (no custom lr scheduler imports needed for cosine)

# Suppress warnings
warnings.filterwarnings("ignore")

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Use all available GPUs (8 GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Configuration
MODEL_NAME = "GSAI-ML/LLaDa-8B-Instruct" # Using Instruct model
DATASET_NAME = "AI-MO/NuminaMath-LEAN"
# Prefer fast local storage if provided via env var FAST_OUTPUT_DIR; defaults to current folder
OUTPUT_DIR = os.environ.get("FAST_OUTPUT_DIR", "./llada-moe-numina-finetuned-30k-5epochs")
MAX_SAMPLES = None  # Use full dataset
TRAIN_SAMPLES = 1000  # Cap training set to 30k samples
MAX_LENGTH = 2048

# Save/IO knobs to control end-of-training CPU work
SAVE_OPTIMIZER_STATE = False      # Large; only needed for resume-training
SAVE_SCHEDULER_STATE = False      # Large; only needed for resume-training
SAVE_TRAINER_STATE = False        # Optional; can be big on long runs
SAVE_TOKENIZER = True             # Small; generally useful

# Variable-length training (paper-inspired): apply short context on ~1% of steps
VARIABLE_LENGTH_TRAINING = True
VARIABLE_LENGTH_PROB = 0.01
VARIABLE_LENGTH_MIN = 8  # min target length when sampling variable length

# No custom LR scheduler configuration needed for cosine

# Disable automatic checkpointing during training
os.environ["TRANSFORMERS_SAVE_LARGE_MODEL_EVERY_N_HOURS"] = "999999"

def log_timing(message):
    """Helper function to log timing information"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

def format_instruction(example, tokenizer):
    """Format the dataset examples into instruction format"""
    formal_statement = example.get('formal_statement', '')
    formal_ground_truth = example.get('formal_ground_truth', '')
    
    # Skip examples without both formal_statement and formal_ground_truth
    if not formal_statement or not formal_ground_truth:
        return {"text": "", "prompt": ""}
    
    # The formal_statement is the input (the theorem to prove)
    instruction = formal_statement
    
    # The formal_ground_truth is the output (the complete Lean proof)
    response = formal_ground_truth
    
    # Using the chat template format
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specialized in Lean theorem proving."},
        {"role": "user", "content": instruction}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full_text = prompt + response + tokenizer.eos_token
    
    return {"text": full_text, "prompt": prompt}

def forward_process_moe(input_ids, mask_id=50256, eps=1e-3):
    """LLaDA-MoE forward process - add masks according to diffusion schedule"""
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)
    return noisy_batch, masked_indices, p_mask

def main():
    log_timing("Starting LLaDA-MoE optimized training script")
    
    # Check GPUs
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Load tokenizer first
    log_timing("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Load model with optimizations
    log_timing(f"Loading model: {MODEL_NAME}")
    model_load_start = time.time()
    
    # OPTIMIZATION: Load model directly on GPUs with auto device map
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",  # Auto distribute across GPUs
        low_cpu_mem_usage=True,
    )
    
    model_load_time = time.time() - model_load_start
    log_timing(f"Model loaded in {model_load_time:.2f} seconds")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    active_params = getattr(model.config, 'num_experts_per_tok', 1) * total_params / getattr(model.config, 'num_experts', 1) if hasattr(model.config, 'num_experts') else total_params
    print(f"Total parameters: {total_params / 1e9:.2f}B")
    print(f"Active parameters during inference: {active_params / 1e9:.2f}B")
    
    # Enable training
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load dataset
    log_timing(f"Loading dataset: {DATASET_NAME}")
    raw_ds = load_dataset(DATASET_NAME, split="train")
    if isinstance(MAX_SAMPLES, int):
        raw_ds = raw_ds.select(range(min(MAX_SAMPLES, len(raw_ds))))
    print(f"Raw dataset size: {len(raw_ds)}")

    # Format + filter
    formatted_ds = raw_ds.map(lambda x: format_instruction(x, tokenizer), remove_columns=raw_ds.column_names)
    formatted_ds = formatted_ds.filter(lambda x: x["text"] != "")
    print(f"Formatted+filtered size: {len(formatted_ds)}")

    # 80/10/10 split on the full formatted dataset
    test_ratio = 0.10
    val_ratio = 0.10
    tv_ratio = test_ratio + val_ratio
    split1 = formatted_ds.train_test_split(test_size=tv_ratio, seed=42)
    remain, tv = split1["train"], split1["test"]
    # Normalize val portion from remaining tv
    remain_val_ratio = val_ratio / tv_ratio
    split2 = tv.train_test_split(test_size=remain_val_ratio, seed=42)
    val_ds, test_ds = split2["train"], split2["test"]

    # Cap training set to 30k samples
    if len(remain) > TRAIN_SAMPLES:
        train_ds = remain.select(range(TRAIN_SAMPLES))
    else:
        train_ds = remain

    print(f"Final splits -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    
    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        
        prompt_tokenized = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        
        tokenized["prompt_lengths"] = [len(ids) for ids in prompt_tokenized["input_ids"]]
        
        return tokenized
    
    log_timing("Tokenizing dataset")
    tokenized_train = train_ds.map(tokenize_function, batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(tokenize_function, batched=True, remove_columns=val_ds.column_names)
    tokenized_test = test_ds.map(tokenize_function, batched=True, remove_columns=test_ds.column_names)
    
    # Custom data collator
    class LLaDAMoEDataCollator:
        def __init__(self, tokenizer, max_length=2048,
                     enable_variable_length=VARIABLE_LENGTH_TRAINING,
                     varlen_prob=VARIABLE_LENGTH_PROB,
                     varlen_min=VARIABLE_LENGTH_MIN):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.enable_variable_length = enable_variable_length
            self.varlen_prob = varlen_prob
            self.varlen_min = varlen_min
            
        def __call__(self, features):
            batch = {}
            
            input_ids = [f["input_ids"] for f in features]
            prompt_lengths = [f["prompt_lengths"] for f in features]
            
            # Base target length = longest in batch (capped by max_length)
            base_len = min(max(len(ids) for ids in input_ids), self.max_length)

            # Optional: sample a shorter target length with small probability (train-time only)
            max_prompt_len = max(int(pl) for pl in prompt_lengths) if prompt_lengths else 0
            max_len = base_len
            if self.enable_variable_length and random.random() < self.varlen_prob:
                sampled = random.randint(self.varlen_min, self.max_length)
                # Never cut below the prompt region in this batch
                max_len = max(min(sampled, self.max_length), max_prompt_len, 1)
            
            padded_input_ids = []
            padded_prompt_lengths = []
            
            for ids, prompt_len in zip(input_ids, prompt_lengths):
                if len(ids) > max_len:
                    ids = ids[:max_len]
                    prompt_len = min(prompt_len, max_len)
                
                padding_length = max_len - len(ids)
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                padded_ids = ids + [pad_id] * padding_length
                
                padded_input_ids.append(padded_ids)
                padded_prompt_lengths.append(prompt_len)
            
            batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
            batch["prompt_lengths"] = torch.tensor(padded_prompt_lengths, dtype=torch.long)
            
            return batch
    
    # We'll toggle variable-length only during training via a callback
    data_collator = LLaDAMoEDataCollator(tokenizer, max_length=MAX_LENGTH,
                                         enable_variable_length=VARIABLE_LENGTH_TRAINING)
    
    # Custom trainer with optimized saving
    class OptimizedLLaDAMoETrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_times = []
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            input_ids = inputs["input_ids"]
            prompt_lengths = inputs["prompt_lengths"]
            
            # LLaDA-MoE forward process
            noisy_batch, masked_indices, p_mask = forward_process_moe(input_ids)
            
            p_mask = torch.clamp(p_mask, min=1e-6, max=1.0)
            
            # Don't mask the prompt
            token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
            prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
            noisy_batch[prompt_mask] = input_ids[prompt_mask]
            
            # Calculate answer lengths
            prompt_mask = prompt_mask.to(torch.int64)
            answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
            answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
            answer_lengths = torch.clamp(answer_lengths, min=1)
            
            # Forward pass
            model_inputs = {"input_ids": noisy_batch, "use_cache": False}
            
            try:
                outputs = model(**model_inputs)
                logits = outputs.logits
                
                # Get auxiliary loss if available (for MoE models)
                aux_loss = getattr(outputs, "aux_loss", 0.0)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                    return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
                else:
                    raise e
            
            # Compute diffusion loss on masked tokens
            mask_id = 50256  # LLaDA-MoE mask token
            masked_indices = (noisy_batch == mask_id)
            
            if masked_indices.sum() > 0:
                token_loss = nn.functional.cross_entropy(
                    logits[masked_indices],
                    input_ids[masked_indices],
                    reduction='none'
                )
                
                token_loss = torch.nan_to_num(token_loss, nan=0.0, posinf=10.0, neginf=0.0)
                token_loss = token_loss / p_mask[masked_indices]
                loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
                
                # Add auxiliary loss for MoE
                if isinstance(aux_loss, torch.Tensor) and aux_loss.numel() > 0:
                    loss = loss + 0.01 * aux_loss  # Weight the aux loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    loss = torch.tensor(1.0, device=input_ids.device, requires_grad=True)
            else:
                loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            
            return (loss, outputs) if return_outputs else loss
        
        def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
            model_inputs = {
                "input_ids": inputs["input_ids"],
                "use_cache": False
            }
            
            with torch.no_grad():
                try:
                    outputs = model(**model_inputs)
                    loss = self.compute_loss(model, inputs)
                    if loss.requires_grad:
                        loss = loss.detach()
                except Exception as e:
                    print(f"Warning: Error in prediction_step: {e}")
                    loss = torch.tensor(float('inf'), device=inputs["input_ids"].device)
            
            return (loss, None, None)
        
        def _save(self, output_dir: str = None, state_dict=None):
            """Optimized save method"""
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Only save on main process
            if self.args.local_rank not in [-1, 0]:
                return
            
            save_start = time.time()
            log_timing(f"Starting optimized save to {output_dir}")
            
            # Get the model to save
            if hasattr(self.model, "module"):
                model_to_save = self.model.module
            else:
                model_to_save = self.model

            # Prefer the library's save_pretrained which can shard and avoid huge single CPU spikes
            try:
                log_timing("Calling model.save_pretrained with safe serialization and sharding...")
                model_to_save.save_pretrained(
                    output_dir,
                    safe_serialization=True,
                    max_shard_size="1GB",
                )
            except Exception as e:
                # Fallback to manual safetensors save if needed
                log_timing(f"save_pretrained failed ({e}); falling back to manual safetensors save")
                # Save configuration
                if hasattr(model_to_save, "config"):
                    model_to_save.config.save_pretrained(output_dir)
                # As a last resort, gather state_dict (can be slow on very large models)
                log_timing("Gathering state dict from model (this may take time)...")
                sd = state_dict if state_dict is not None else model_to_save.state_dict()
                keys_to_remove = [k for k in sd.keys() if "_float_tensor" in k]
                for k in keys_to_remove:
                    del sd[k]
                log_timing("Saving model weights using safetensors (manual path)...")
                save_file(sd, os.path.join(output_dir, "model.safetensors"))

            # Optionally save training state (skip by default to avoid heavy CPU/disk IO)
            if SAVE_OPTIMIZER_STATE and hasattr(self, "optimizer") and self.optimizer is not None:
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            if SAVE_SCHEDULER_STATE and hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            if SAVE_TRAINER_STATE:
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            # Save tokenizer
            if SAVE_TOKENIZER and self.tokenizer is not None and self.is_world_process_zero():
                self.tokenizer.save_pretrained(output_dir)
            
            save_time = time.time() - save_start
            self.save_times.append(save_time)
            log_timing(f"Save completed in {save_time:.2f} seconds")

        # Use default Trainer scheduler creation (cosine via TrainingArguments)
    
    # Callback for metrics
    class MetricsCallback(TrainerCallback):
        def __init__(self):
            self.training_logs = []
            
        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            if logs:
                self.training_logs.append({
                    "step": state.global_step,
                    "epoch": state.epoch,
                    **logs
                })
                print(f"Step {state.global_step}: {logs}")
    
    metrics_callback = MetricsCallback()
    
    # Callback to disable variable-length during evaluation and re-enable during training
    class VariableLengthToggleCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if isinstance(self, TrainerCallback):
                if hasattr(data_collator, 'enable_variable_length'):
                    data_collator.enable_variable_length = VARIABLE_LENGTH_TRAINING
        def on_evaluate(self, args, state, control, **kwargs):
            if hasattr(data_collator, 'enable_variable_length'):
                data_collator.enable_variable_length = False
        def on_evaluate_end(self, args, state, control, **kwargs):
            if hasattr(data_collator, 'enable_variable_length'):
                data_collator.enable_variable_length = VARIABLE_LENGTH_TRAINING
        def on_predict(self, args, state, control, **kwargs):
            if hasattr(data_collator, 'enable_variable_length'):
                data_collator.enable_variable_length = False
        def on_prediction_end(self, args, state, control, **kwargs):
            if hasattr(data_collator, 'enable_variable_length'):
                data_collator.enable_variable_length = VARIABLE_LENGTH_TRAINING
    
    # Training arguments optimized for fast saving
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Effective batch size = 1 * 8 * 4 = 32
    learning_rate=5e-5,  # Peak LR for cosine schedule
        bf16=True,
        # Save only at the very end (we call trainer.save_model manually)
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=1000,
        logging_steps=10,
    warmup_steps=50,
        optim="adamw_torch",
        save_total_limit=1,  # Keep for safety if save is re-enabled
        load_best_model_at_end=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=False,  # Disable for MoE models
        weight_decay=0.01,
    lr_scheduler_type="cosine",
        # Performance settings
        tf32=True,
        dataloader_pin_memory=True,
        # Distributed settings
        ddp_timeout=3600,  # 1 hour timeout
        dataloader_drop_last=True,
        save_on_each_node=False,
        save_safetensors=True,  # Use safetensors format
    )
    
    # Initialize trainer
    trainer = OptimizedLLaDAMoETrainer(
        model=model,
        args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[metrics_callback, VariableLengthToggleCallback()],
        tokenizer=tokenizer,
    )
    
    log_timing("Starting training...")
    
    try:
        trainer.train()
        
        # Report average save time
        if trainer.save_times:
            avg_save_time = sum(trainer.save_times) / len(trainer.save_times)
            print(f"\nAverage checkpoint save time: {avg_save_time:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_model(OUTPUT_DIR + "_interrupted")
        tokenizer.save_pretrained(OUTPUT_DIR + "_interrupted")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        try:
            trainer.save_model(OUTPUT_DIR + "_error")
            tokenizer.save_pretrained(OUTPUT_DIR + "_error")
        except:
            pass
        raise
    
    log_timing(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model()
    if SAVE_TOKENIZER:
        tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save metrics
    metrics_file = os.path.join(OUTPUT_DIR, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_callback.training_logs, f, indent=2)
    
    # Create plots
    def plot_training_metrics(logs, output_dir):
        # Per-step training loss typically appears under key 'loss'
        train_step_logs = [log for log in logs if "loss" in log and "step" in log]
        # Aggregated final train loss (one entry) sometimes appears under 'train_loss'
        train_final_logs = [log for log in logs if "train_loss" in log and "step" in log]
        eval_logs = [log for log in logs if "eval_loss" in log and "step" in log]

        has_any = bool(train_step_logs or train_final_logs or eval_logs)
        if not has_any:
            return

        # Determine layout: one subplot if only one series, else two
        two_panels = bool((train_step_logs or train_final_logs) and eval_logs)
        plt.figure(figsize=(12, 5))

        # Training panel
        if train_step_logs or train_final_logs:
            if two_panels:
                ax1 = plt.subplot(1, 2, 1)
            else:
                ax1 = plt.gca()

            if train_step_logs:
                steps = [log["step"] for log in train_step_logs]
                losses = [log["loss"] for log in train_step_logs]
                ax1.plot(steps, losses, 'b-', marker='o', markersize=3, label='Training Loss')

            if train_final_logs:
                f_steps = [log["step"] for log in train_final_logs]
                f_losses = [log["train_loss"] for log in train_final_logs]
                ax1.plot(f_steps, f_losses, 'bx', markersize=6, label='Final Train Loss')

            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss (LLaDA-MoE)')
            ax1.grid(True)
            ax1.legend()

        # Eval panel
        if eval_logs:
            ax2 = plt.subplot(1, 2, 2) if two_panels else plt.gca()
            eval_steps = [log["step"] for log in eval_logs]
            eval_values = [log["eval_loss"] for log in eval_logs]
            ax2.plot(eval_steps, eval_values, 'r-', marker='s', markersize=3, label='Validation Loss')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Loss')
            ax2.set_title('Validation Loss (LLaDA-MoE)')
            ax2.grid(True)
            ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_plots.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    log_timing("Rendering training plots...")
    plot_training_metrics(metrics_callback.training_logs, OUTPUT_DIR)
    log_timing("Plots saved.")
    
    # Save configuration
    config = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "max_samples": MAX_SAMPLES,
        "max_length": MAX_LENGTH,
        "training_type": "llada_moe_optimized",
        "total_parameters": f"{total_params / 1e9:.2f}B",
        "active_parameters": f"{active_params / 1e9:.2f}B",
        "trainable_parameters": trainable_params,
        "optimizations": [
            "Using LLaDA-MoE model (7B total, 1.4B active)",
            "Safetensors format for fast saving",
            "Save on main process only",
            "Reduced checkpoint frequency",
            "Direct device_map='auto' loading",
        ],
        "average_save_time": sum(trainer.save_times) / len(trainer.save_times) if trainer.save_times else "N/A"
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    log_timing("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Training metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()
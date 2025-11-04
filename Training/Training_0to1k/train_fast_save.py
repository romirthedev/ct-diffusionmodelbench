import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
import transformers
from accelerate import Accelerator
import warnings
from safetensors.torch import save_file
import shutil

# Suppress warnings
warnings.filterwarnings("ignore")

# Set environment variables to fix tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Use all available GPUs (8 GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Configuration
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
DATASET_NAME = "AI-MO/NuminaMath-LEAN"
OUTPUT_DIR = "./llada-numina-finetuned-1k-optimized"
MAX_SAMPLES = 100  # Quick test with 100 samples
MAX_LENGTH = 2048  # Further reduced to prevent OOM on multi-GPU

# OPTIMIZATION: Disable automatic checkpointing during training
os.environ["TRANSFORMERS_SAVE_LARGE_MODEL_EVERY_N_HOURS"] = "999999"

def format_instruction(example):
    """Format the dataset examples into instruction format for NuminaMath-LEAN with prompt tracking."""
    problem = example.get('problem', '')
    formal_statement = example.get('formal_statement', '')
    formal_proof = example.get('formal_proof', '')
    answer = example.get('answer', '')
    
    if formal_statement:
        instruction = f"Problem: {problem}\n\nFormal Statement: {formal_statement}"
    else:
        instruction = f"Problem: {problem}"
    
    response = formal_proof if formal_proof else answer
    
    if not response:
        return {"text": "", "prompt": ""}
    
    prompt_text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    full_text = prompt_text + response + "<|eot_id|>"
    
    return {"text": full_text, "prompt": prompt_text}

def forward_process(input_ids, eps=1e-3):
    """LLaDA forward process - add masks according to diffusion schedule"""
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask

def main():
    print("Loading tokenizer and model...")
    
    # Initialize accelerator with optimizations
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=2,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # OPTIMIZATION 1: Load model on single GPU first, then distribute
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Load model on CPU first to avoid distributed loading issues
    print("Loading model on CPU first...")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,  # Load on CPU first
        low_cpu_mem_usage=True,
    )
    
    # OPTIMIZATION 2: Use accelerator to handle distribution
    print("Distributing model across GPUs using accelerator...")
    model = accelerator.prepare_model(model)
    
    # Enable full fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"Loading dataset: {DATASET_NAME}")
    
    # Load dataset
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    
    print(f"Dataset size: {len(dataset)}")
    
    # Format dataset
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Split into train and validation
    dataset = dataset.train_test_split(test_size=0.15, seed=42)
    
    # Tokenize function with prompt length tracking
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
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Custom data collator for LLaDA diffusion training
    class LLaDADataCollator:
        def __init__(self, tokenizer, max_length=1024):
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __call__(self, features):
            batch = {}
            
            input_ids = [f["input_ids"] for f in features]
            prompt_lengths = [f["prompt_lengths"] for f in features]
            
            max_len = min(max(len(ids) for ids in input_ids), self.max_length)
            
            padded_input_ids = []
            padded_prompt_lengths = []
            
            for ids, prompt_len in zip(input_ids, prompt_lengths):
                if len(ids) > max_len:
                    ids = ids[:max_len]
                    prompt_len = min(prompt_len, max_len)
                
                padding_length = max_len - len(ids)
                padded_ids = ids + [self.tokenizer.eos_token_id] * padding_length
                
                padded_input_ids.append(padded_ids)
                padded_prompt_lengths.append(prompt_len)
            
            batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
            batch["prompt_lengths"] = torch.tensor(padded_prompt_lengths, dtype=torch.long)
            
            return batch
    
    data_collator = LLaDADataCollator(tokenizer, max_length=MAX_LENGTH)
    
    # OPTIMIZATION 3: Custom trainer with fast saving
    class FastSaveLLaDATrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            input_ids = inputs["input_ids"]
            prompt_lengths = inputs["prompt_lengths"]
            
            noisy_batch, masked_indices, p_mask = forward_process(input_ids)
            
            p_mask = torch.clamp(p_mask, min=1e-6, max=1.0)
            
            token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
            prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
            noisy_batch[prompt_mask] = input_ids[prompt_mask]
            
            prompt_mask = prompt_mask.to(torch.int64)
            answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
            answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
            
            answer_lengths = torch.clamp(answer_lengths, min=1)
            
            model_inputs = {"input_ids": noisy_batch, "use_cache": False}
            
            try:
                outputs = model(**model_inputs)
                logits = outputs.logits
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                    return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
                else:
                    raise e
            
            masked_indices = (noisy_batch == 126336)
            
            if masked_indices.sum() > 0:
                token_loss = nn.functional.cross_entropy(
                    logits[masked_indices],
                    input_ids[masked_indices],
                    reduction='none'
                )
                
                token_loss = torch.nan_to_num(token_loss, nan=0.0, posinf=10.0, neginf=0.0)
                token_loss = token_loss / p_mask[masked_indices]
                loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
                
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
            """OPTIMIZED: Custom save method that avoids gathering distributed model"""
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # OPTIMIZATION 4: Only save on main process
            if self.args.local_rank not in [-1, 0]:
                return
            
            print(f"Fast saving model to {output_dir}...")
            
            # OPTIMIZATION 5: Save model shards instead of gathering
            if hasattr(self.model, "module"):
                # For DataParallel or DistributedDataParallel
                model_to_save = self.model.module
            else:
                model_to_save = self.model
            
            # OPTIMIZATION 6: Save configuration first
            if hasattr(model_to_save, "config"):
                model_to_save.config.save_pretrained(output_dir)
            
            # OPTIMIZATION 7: Save model weights using safetensors (faster)
            if state_dict is None:
                state_dict = model_to_save.state_dict()
            
            # Remove any keys that shouldn't be saved
            keys_to_remove = []
            for k in state_dict.keys():
                if "_float_tensor" in k:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del state_dict[k]
            
            # Save using safetensors
            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
            
            # Save training state
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            
            # Save trainer state
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
            
            # Save tokenizer
            if self.tokenizer is not None and self.is_world_process_zero():
                self.tokenizer.save_pretrained(output_dir)
            
            print(f"Model saved successfully to {output_dir}")
    
    # Custom callback to track training metrics
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
    
    # Training arguments optimized for fast saving
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        bf16=True,
        save_strategy="steps",
        save_steps=50,  # Less frequent saves
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        warmup_steps=100,
        optim="adamw_torch",
        save_total_limit=1,  # Only keep latest checkpoint
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        # Performance settings
        tf32=True,
        dataloader_pin_memory=False,
        fp16=False,
        # Distributed training settings
        ddp_timeout=1800,
        dataloader_drop_last=True,
        eval_accumulation_steps=1,
        eval_delay=0,
        save_on_each_node=False,
        # OPTIMIZATION: Save safetensors format
        save_safetensors=True,
    )
    
    # Initialize trainer
    trainer = FastSaveLLaDATrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        callbacks=[metrics_callback],
        tokenizer=tokenizer,
    )
    
    print("Starting optimized training...")
    
    try:
        trainer.train()
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
    
    print(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training metrics
    metrics_file = os.path.join(OUTPUT_DIR, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_callback.training_logs, f, indent=2)
    
    # Create loss plots
    def plot_training_metrics(logs, output_dir):
        train_losses = [log for log in logs if "train_loss" in log]
        eval_losses = [log for log in logs if "eval_loss" in log]
        
        if train_losses:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            steps = [log["step"] for log in train_losses]
            losses = [log["train_loss"] for log in train_losses]
            plt.plot(steps, losses, 'b-', label='Training Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.legend()
            plt.grid(True)
            
            if eval_losses:
                plt.subplot(2, 2, 2)
                eval_steps = [log["step"] for log in eval_losses]
                eval_loss_values = [log["eval_loss"] for log in eval_losses]
                plt.plot(eval_steps, eval_loss_values, 'r-', label='Validation Loss')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Validation Loss Over Time')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(2, 2, 3)
                plt.plot(steps, losses, 'b-', label='Training Loss')
                plt.plot(eval_steps, eval_loss_values, 'r-', label='Validation Loss')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Training vs Validation Loss')
                plt.legend()
                plt.grid(True)
            
            lr_logs = [log for log in logs if "learning_rate" in log]
            if lr_logs:
                plt.subplot(2, 2, 4)
                lr_steps = [log["step"] for log in lr_logs]
                lr_values = [log["learning_rate"] for log in lr_logs]
                plt.plot(lr_steps, lr_values, 'g-', label='Learning Rate')
                plt.xlabel('Steps')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_plots.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Training plots saved to: {os.path.join(output_dir, 'training_plots.png')}")
    
    plot_training_metrics(metrics_callback.training_logs, OUTPUT_DIR)
    
    # Save configuration
    config = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "max_samples": MAX_SAMPLES,
        "max_length": MAX_LENGTH,
        "training_type": "full_fine_tuning_optimized",
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizations": [
            "Load model on CPU first then distribute",
            "Use accelerator for distribution",
            "Custom fast save method",
            "Save on main process only",
            "Save model shards instead of gathering",
            "Use safetensors format",
            "Reduced checkpoint frequency",
        ]
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("\nOptimized training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Training metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()
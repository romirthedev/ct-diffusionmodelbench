import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
import transformers

# Configuration
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
DATASET_NAME = "AI-MO/NuminaMath-LEAN"
OUTPUT_DIR = "./llada-numina-finetuned"
MAX_SAMPLES = 1000
MAX_LENGTH = 1024  # Reduced to prevent OOM

def format_instruction(example):
    """Format the dataset examples into instruction format for NuminaMath-LEAN."""
    # Use 'problem' as the main instruction, with formal_statement as additional context
    problem = example.get('problem', '')
    formal_statement = example.get('formal_statement', '')
    formal_proof = example.get('formal_proof', '')
    answer = example.get('answer', '')
    
    # Create instruction from problem and formal statement
    if formal_statement:
        instruction = f"Problem: {problem}\n\nFormal Statement: {formal_statement}"
    else:
        instruction = f"Problem: {problem}"
    
    # Use formal_proof if available, otherwise use answer
    response = formal_proof if formal_proof else answer
    
    # Skip examples without proper response
    if not response:
        return {"text": ""}
    
    # Create a chat template format
    formatted_text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""
    
    return {"text": formatted_text}

def main():
    print("Loading tokenizer and model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model for full fine-tuning on A100
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Enable full fine-tuning - all parameters are trainable
    for param in model.parameters():
        param.requires_grad = True
    
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"Loading dataset: {DATASET_NAME}")
    
    # Load dataset
    dataset = load_dataset(DATASET_NAME, split="train")
    
    # Take first 1000 samples
    dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample example: {dataset[0]}")
    
    # Format dataset
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Split into train and validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Custom trainer class to handle loss computation
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Compute custom loss
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            else:
                loss = None
            
            return (loss, outputs) if return_outputs else loss

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
    
    metrics_callback = MetricsCallback()
    
    # Training arguments optimized for full fine-tuning on A100
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Further reduced to prevent OOM
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,  # Reduced to prevent OOM
        learning_rate=5e-5,  # Lower learning rate for full fine-tuning
        bf16=True,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        warmup_steps=100,
        optim="adamw_torch",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
    )
    
    # Initialize custom trainer with metrics callback
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        callbacks=[metrics_callback],
    )
    
    print("Starting full fine-tuning...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training metrics to JSON
    metrics_file = os.path.join(OUTPUT_DIR, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_callback.training_logs, f, indent=2)
    
    # Create loss plots
    def plot_training_metrics(logs, output_dir):
        # Extract training and validation losses
        train_losses = [log for log in logs if "train_loss" in log]
        eval_losses = [log for log in logs if "eval_loss" in log]
        
        if train_losses:
            plt.figure(figsize=(12, 8))
            
            # Plot training loss
            plt.subplot(2, 2, 1)
            steps = [log["step"] for log in train_losses]
            losses = [log["train_loss"] for log in train_losses]
            plt.plot(steps, losses, 'b-', label='Training Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.legend()
            plt.grid(True)
            
            # Plot evaluation loss
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
                
                # Plot both losses together
                plt.subplot(2, 2, 3)
                plt.plot(steps, losses, 'b-', label='Training Loss')
                plt.plot(eval_steps, eval_loss_values, 'r-', label='Validation Loss')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Training vs Validation Loss')
                plt.legend()
                plt.grid(True)
            
            # Plot learning rate if available
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
    
    # Generate plots
    plot_training_metrics(metrics_callback.training_logs, OUTPUT_DIR)
    
    # Save training configuration
    config = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "max_samples": MAX_SAMPLES,
        "max_length": MAX_LENGTH,
        "training_type": "full_fine_tuning",
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Training metrics saved to: {metrics_file}")
    print(f"Training configuration saved for easy resumption.")
    print(f"Total parameters trained: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

if __name__ == "__main__":
    main()
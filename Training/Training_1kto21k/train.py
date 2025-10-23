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

# Configuration - Updated to use finetuned model and train from row 1k for 20k more rows
MODEL_NAME = "../llada-numina-finetuned"  # Use the already finetuned model
DATASET_NAME = "AI-MO/NuminaMath-LEAN"
OUTPUT_DIR = "./llada-numina-continued-1kto21k"
START_ROW = 1000  # Start from row 1k
MAX_SAMPLES = 20000  # Train for 20k more rows
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
    
    # Check if finetuned model exists
    if not os.path.exists(MODEL_NAME):
        print(f"Error: Finetuned model not found at {MODEL_NAME}")
        print("Please ensure the finetuned model exists before running continued training.")
        return
    
    # Load tokenizer from finetuned model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load the already finetuned model for continued training
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
    print(f"Starting from finetuned model: {MODEL_NAME}")
    
    print(f"Loading dataset: {DATASET_NAME}")
    
    # Load dataset
    dataset = load_dataset(DATASET_NAME, split="train")
    
    # Select rows from START_ROW to START_ROW + MAX_SAMPLES
    end_row = min(START_ROW + MAX_SAMPLES, len(dataset))
    dataset = dataset.select(range(START_ROW, end_row))
    
    print(f"Dataset size: {len(dataset)} (rows {START_ROW} to {end_row-1})")
    print(f"Sample example: {dataset[0]}")
    
    # Format dataset
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Split into train and validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Tokenize function
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_tensors=None,  # Return Python lists, not tensors
        )
        # Ensure input_ids are integers
        if 'input_ids' in tokenized:
            tokenized['input_ids'] = [[int(token_id) for token_id in seq] for seq in tokenized['input_ids']]
        if 'attention_mask' in tokenized:
            tokenized['attention_mask'] = [[int(mask) for mask in seq] for seq in tokenized['attention_mask']]
        return tokenized
    
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
            # Ensure input_ids are LongTensor
            if 'input_ids' in inputs:
                inputs['input_ids'] = inputs['input_ids'].long()
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].long()
            if 'labels' in inputs:
                inputs['labels'] = inputs['labels'].long()
                
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
    
    # Training arguments optimized for continued fine-tuning
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Further reduced to prevent OOM
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,  # Reduced to prevent OOM
        learning_rate=3e-5,  # Slightly lower learning rate for continued training
        bf16=True,
        save_strategy="steps",
        save_steps=100,  # Save more frequently for longer training
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=20,
        warmup_steps=200,  # More warmup steps for larger dataset
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
    
    print(f"Starting continued fine-tuning from row {START_ROW} for {MAX_SAMPLES} samples...")
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
        "base_model_name": "GSAI-ML/LLaDA-8B-Instruct",
        "finetuned_model_path": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "start_row": START_ROW,
        "max_samples": MAX_SAMPLES,
        "max_length": MAX_LENGTH,
        "training_type": "continued_fine_tuning",
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("Continued training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Training metrics saved to: {metrics_file}")
    print(f"Training configuration saved for easy resumption.")
    print(f"Total parameters trained: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Trained on rows {START_ROW} to {end_row-1} ({len(dataset['train']) + len(dataset['test'])} samples)")

if __name__ == "__main__":
    main()
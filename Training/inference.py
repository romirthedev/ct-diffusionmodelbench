import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
FINETUNED_MODEL_PATH = "./llada-numina-finetuned"

def load_model():
    """Load the fully fine-tuned model."""
    print("Loading fine-tuned model and tokenizer...")
    
    # Load the fine-tuned model directly
    model = AutoModelForCausalLM.from_pretrained(
        FINETUNED_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    
    # Load training metrics if available
    metrics_file = os.path.join(FINETUNED_MODEL_PATH, "training_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            training_metrics = json.load(f)
        print(f"Training completed with {len(training_metrics)} logged steps")
        
        # Show final metrics
        final_metrics = training_metrics[-1] if training_metrics else {}
        if 'eval_loss' in final_metrics:
            print(f"Final validation loss: {final_metrics['eval_loss']:.4f}")
        if 'train_loss' in final_metrics:
            print(f"Final training loss: {final_metrics['train_loss']:.4f}")
    
    return model, tokenizer

def generate_response(model, tokenizer, instruction, max_length=512):
    """Generate a response for a given instruction."""
    
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    
    return response

def main():
    model, tokenizer = load_model()
    
    print("\n" + "="*50)
    print("Fine-tuned Model Inference")
    print("="*50 + "\n")
    
    # Example test cases - mathematical problems similar to NuminaMath-LEAN
    test_instructions = [
        "Prove that the sum of two even numbers is even.",
        "What is the derivative of x^2 + 3x + 5?",
        "Solve for x: 2x + 5 = 15",
        "Find the limit of (x^2 - 1)/(x - 1) as x approaches 1.",
        "Prove that if n is odd, then n^2 is odd.",
    ]
    
    for i, instruction in enumerate(test_instructions, 1):
        print(f"\nTest {i}:")
        print(f"Instruction: {instruction}")
        print(f"Response: {generate_response(model, tokenizer, instruction)}")
        print("-" * 50)
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*50 + "\n")
    
    while True:
        instruction = input("\nEnter your instruction: ")
        if instruction.lower() in ['quit', 'exit', 'q']:
            break
        
        response = generate_response(model, tokenizer, instruction)
        print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import json
import time
from datetime import datetime

# Configuration
MODEL_NAME = "./llada-moe-numina-finetuned-optimized"  # Use the fine-tuned model
DATASET_NAME = "AI-MO/NuminaMath-LEAN"
NUM_SAMPLES = 10
OUTPUT_FILE = "llada_moe_finetuned_test_results.json"

def log_timing(message):
    """Helper function to log timing information"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=156895):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

def format_problem_for_inference(problem_text):
    """Format a problem for the LLaDA-MoE model"""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specialized in mathematical reasoning and formal proofs."},
        {"role": "user", "content": f"Problem: {problem_text}\n\nPlease provide a step-by-step solution."}
    ]
    return messages

def main():
    log_timing("Starting LLaDA-MoE test script")
    
    # Check available device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        print("WARNING: Running on CPU - this will be slow!")
    
    log_timing(f"Loading model: {MODEL_NAME}")
    try:
        # Load model with proper settings for MoE
        model = AutoModel.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Automatically distribute across available GPUs
            low_cpu_mem_usage=True,
        ).eval()
        
        log_timing("Model loaded successfully")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total model parameters: {total_params / 1e9:.2f}B")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        
        # Alternative: Load on single GPU
        model = AutoModel.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
    
    # Load tokenizer
    log_timing("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Load dataset
    log_timing(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    # Select samples
    samples = dataset.select(range(min(NUM_SAMPLES, len(dataset))))
    print(f"Testing on {len(samples)} samples")
    
    results = []
    
    for idx, sample in enumerate(samples):
        log_timing(f"Processing sample {idx + 1}/{len(samples)}")
        
        problem = sample.get('problem', '')
        formal_statement = sample.get('formal_statement', '')
        expected_answer = sample.get('answer', '')
        expected_proof = sample.get('formal_proof', '')
        
        if not problem:
            print(f"Skipping sample {idx} - no problem text")
            continue
        
        # Format the problem
        messages = format_problem_for_inference(problem)
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Tokenize
        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        
        print(f"\nProblem {idx + 1}: {problem[:100]}...")
        
        # Time the generation
        start_time = time.time()
        
        try:
            # Generate using the diffusion method
            with torch.no_grad():
                generated_ids = generate(
                    model, 
                    input_ids, 
                    steps=128, 
                    gen_length=128, 
                    block_length=32, 
                    temperature=0., 
                    cfg_scale=0., 
                    remasking='low_confidence'
                )
            
            # Decode the generated text
            generated_text = tokenizer.batch_decode(
                generated_ids[:, input_ids.shape[1]:], 
                skip_special_tokens=False
            )[0]
            
            generation_time = time.time() - start_time
            
            # Store result
            result = {
                "index": idx,
                "problem": problem,
                "formal_statement": formal_statement,
                "expected_answer": expected_answer,
                "expected_proof": expected_proof,
                "generated_response": generated_text,
                "generation_time_seconds": generation_time,
                "input_length": input_ids.shape[1],
                "output_length": generated_ids.shape[1] - input_ids.shape[1],
            }
            
            results.append(result)
            
            print(f"Generated response: {generated_text[:200]}...")
            print(f"Generation time: {generation_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error generating response for sample {idx}: {e}")
            result = {
                "index": idx,
                "problem": problem,
                "error": str(e),
            }
            results.append(result)
    
    # Save results
    log_timing(f"Saving results to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total samples processed: {len(results)}")
    successful = [r for r in results if "generated_response" in r]
    print(f"Successful generations: {len(successful)}")
    
    if successful:
        avg_time = sum(r["generation_time_seconds"] for r in successful) / len(successful)
        avg_input_len = sum(r["input_length"] for r in successful) / len(successful)
        avg_output_len = sum(r["output_length"] for r in successful) / len(successful)
        
        print(f"Average generation time: {avg_time:.2f} seconds")
        print(f"Average input length: {avg_input_len:.0f} tokens")
        print(f"Average output length: {avg_output_len:.0f} tokens")
    
    print(f"\nResults saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
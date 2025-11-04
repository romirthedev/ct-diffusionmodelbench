"""
Simple test script for LLaDA model - no input required.
Tests with a fixed Lean4 math problem.
"""

import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

def resolve_mask_id(model, tokenizer) -> int:
    mask_id = getattr(model.config, 'mask_token_id', None)
    if mask_id is None:
        mask_id = getattr(tokenizer, 'mask_token_id', None)
    if mask_id is None and getattr(tokenizer, 'mask_token', None):
        try:
            mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        except Exception:
            mask_id = None

    if mask_id is None or mask_id >= getattr(model.config, 'vocab_size', 10**9):
        for cand in ['<|mask|>', '<mask>', '[MASK]', '<MASK>']:
            try:
                cid = tokenizer.convert_tokens_to_ids(cand)
                if cid is not None and cid != tokenizer.unk_token_id and cid < model.config.vocab_size:
                    mask_id = cid
                    break
            except Exception:
                continue

    if mask_id is None:
        raise ValueError("Could not resolve a valid mask token id.")

    return mask_id

def simple_generate(model, prompt_ids, tokenizer, mask_id, max_tokens=100):
    """Simple diffusion generation - just fill masks left to right"""
    device = model.device
    batch_size = prompt_ids.shape[0]
    
    # Create sequence: prompt + masked tokens
    x = torch.full(
        (batch_size, prompt_ids.shape[1] + max_tokens),
        mask_id,
        dtype=torch.long,
        device=device
    )
    x[:, :prompt_ids.shape[1]] = prompt_ids.clone()
    
    prompt_len = prompt_ids.shape[1]
    attention_mask = torch.ones_like(x, dtype=torch.long, device=device)
    
    print(f"[Info] Generating {max_tokens} tokens...")
    
    # Simple generation: fill masks one by one from left to right
    for pos in range(prompt_len, x.shape[1]):
        print(f"[Step] Filling position {pos - prompt_len + 1}/{max_tokens}")
        
        with torch.no_grad():
            outputs = model(x, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits.float()
        
        # Get logits for current position
        logits_row = logits[0, pos, :]
        
        # Remove mask token from options
        logits_row[mask_id] = -1e9
        
        # Get most likely token (greedy)
        next_token = torch.argmax(logits_row).item()
        x[0, pos] = next_token
        
        # Check if it's EOS token
        if next_token == tokenizer.eos_token_id:
            print(f"[Info] EOS token generated, stopping at position {pos - prompt_len + 1}")
            break
    
    # Decode response
    generated_ids = x[0, prompt_len:]
    
    # Find actual end (before any remaining masks)
    end_pos = max_tokens
    for i, token_id in enumerate(generated_ids):
        if token_id == mask_id:
            end_pos = i
            break
    
    response = tokenizer.decode(generated_ids[:end_pos], skip_special_tokens=True)
    return response.strip()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using: {device}")
    
    # Load model and tokenizer
    model_path = 'GSAI-ML/LLaDA-8B-Instruct'
    print(f"[Loading] Model: {model_path}")
    
    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32
        ).to(device).eval()
        print("[Success] Model loaded")
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        print("[Success] Tokenizer loaded")
    except Exception as e:
        print(f"[Error] Failed to load tokenizer: {e}")
        return
    
    # Get mask token
    try:
        mask_id = resolve_mask_id(model, tokenizer)
        print(f"[Info] Mask token ID: {mask_id}")
    except Exception as e:
        print(f"[Error] Could not resolve mask token: {e}")
        return
    
    # Test prompt - Lean4 math problem
    test_prompt = """<|startoftext|><|start_header_id|>user<|end_header_id|>

Solve this theorem in Lean4: Prove that for any natural number n, n + 0 = n<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    print("=" * 70)
    print("TEST PROMPT:")
    print(test_prompt)
    print("=" * 70)
    
    # Tokenize prompt
    try:
        input_ids = tokenizer(test_prompt, return_tensors="pt")['input_ids'].to(device)
        print(f"[Info] Prompt length: {input_ids.shape[1]} tokens")
    except Exception as e:
        print(f"[Error] Tokenization failed: {e}")
        return
    
    # Generate response
    print("\n[Generating] Starting generation...")
    try:
        response = simple_generate(model, input_ids, tokenizer, mask_id, max_tokens=150)
        
        print("\n" + "=" * 70)
        print("GENERATED RESPONSE:")
        print("=" * 70)
        print(response)
        print("=" * 70)
        
    except Exception as e:
        print(f"[Error] Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" LLaDA Simple Test - Lean4 Problem ")
    print("=" * 70 + "\n")
    main()
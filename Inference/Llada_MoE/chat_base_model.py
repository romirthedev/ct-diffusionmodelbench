"""
Simple chat interface testing LLaDA base model with standard generation.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def chat():
    """
    Interactive chat with LLaDA base model using standard generation.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using: {device}")

    # Use base model
    model_path = 'GSAI-ML/LLaDA-8B-Instruct'
    print(f"[Model] Using base model: {model_path}")

    # Load model - FIXED: Use AutoModelForCausalLM for generation
    print("[Loading] Model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=(torch.bfloat16 if device == 'cuda' else torch.float32),
            device_map="auto"
        )
        model.eval()
        print("[Success] Model loaded")
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    # Load tokenizer
    print("[Loading] Tokenizer...")
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

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Show model info
    print("\n" + "=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)
    print(f"Model path:            {model_path}")
    print(f"Model vocab size:      {model.config.vocab_size}")
    print(f"Tokenizer vocab size:  {tokenizer.vocab_size}")
    print(f"EOS token ID:          {tokenizer.eos_token_id}")
    print(f"PAD token ID:          {tokenizer.pad_token_id}")
    print(f"BOS token ID:          {tokenizer.bos_token_id}")
    print("=" * 70 + "\n")

    # Generation parameters (conservative settings)
    gen_params = {
        'max_new_tokens': 256,
        'temperature': 0.7,
        'do_sample': True,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    print("*" * 70)
    print("GENERATION SETTINGS")
    print("*" * 70)
    for key, value in gen_params.items():
        print(f"  {key}: {value}")
    print("*" * 70)
    print("Type 'quit' to exit\n")

    # Chat loop
    conversation = []
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q", "bye"]:
            print("Goodbye!")
            break

        # Add user message to conversation
        conversation.append({"role": "user", "content": user_input})

        # Create prompt using chat template
        try:
            prompt_text = tokenizer.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
        except Exception as e:
            print(f"[Warning] Chat template failed: {e}")
            # Fallback to simple format
            prompt_text = f"User: {user_input}\nAssistant:"

        print(f"[Debug] Prompt: {prompt_text[:200]}...")

        # Tokenize input
        try:
            input_ids = tokenizer(prompt_text, return_tensors="pt")['input_ids'].to(device)
            print(f"[Info] Input length: {input_ids.shape[1]} tokens")
        except Exception as e:
            print(f"[Error] Tokenization failed: {e}")
            continue

        # Generate response
        print("[Generating]...")
        try:
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    **gen_params
                )
            
            # Decode response
            response_ids = output[0, input_ids.shape[1]:]
            answer = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            
            print("\n" + "=" * 70)
            print("Assistant:", answer)
            print("=" * 70 + "\n")
            
            # Add to conversation
            conversation.append({"role": "assistant", "content": answer})
            
            # Keep conversation manageable
            if len(conversation) > 10:
                conversation = conversation[-10:]
                
        except Exception as e:
            print(f"[Error] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" LLaDA Base Model Chat Test ")
    print("=" * 70 + "\n")
    try:
        chat()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        import traceback
        traceback.print_exc()
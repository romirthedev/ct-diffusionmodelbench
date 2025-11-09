import os
import json
import time
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def add_gumbel_noise(logits: torch.Tensor, temperature: float):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def llada_generate(
    model,
    prompt_ids: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int = 156895,
    avoid_eos: bool = False,
    eos_token_id: Optional[int] = None,
):
    """Diffusion-style masked token generation for LLaDA(-MoE) models.

    prompt_ids: [1, seq_len] on model.device, dtype long
    Returns: [1, seq_len + gen_length] token ids
    """

    x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_ids.shape[1]] = prompt_ids.clone()
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_slice = slice(prompt_ids.shape[1] + num_block * block_length, prompt_ids.shape[1] + (num_block + 1) * block_length)
        block_mask_index = (x[:, block_slice] == mask_id)
        num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # Optionally discourage EOS during generation
            if avoid_eos and eos_token_id is not None:
                logits[..., eos_token_id] = float('-inf')

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = torch.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # prevent advancing beyond current block
            x0_p[:, prompt_ids.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def build_prompt(tokenizer: AutoTokenizer, user_text: str, lean_only: bool = True) -> str:
    sys_content = "You are a helpful, general-purpose AI assistant."
    if lean_only:
        sys_content += " Respond only with Lean code (import Mathlib, theorem, proof). Do not include explanations or natural language."

    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt


def run_chat(
    model_dir: str,
    prompt_text: str,
    max_length: int = 2048,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    avoid_eos: bool = True,
    truncate_at_eos: bool = True,
    lean_only: bool = True,
    mask_id_override: Optional[int] = None,
):
    log(f"Loading tokenizer/model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    # Determine mask token id
    mask_id = mask_id_override
    if mask_id is None:
        mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        # Default for LLaDA-MoE; override via --mask-id for other variants
        mask_id = 156895

    prompt = build_prompt(tokenizer, prompt_text, lean_only=lean_only)
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = toks["input_ids"].to(model.device)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = llada_generate(
            model,
            input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking='low_confidence',
            mask_id=mask_id,
            avoid_eos=avoid_eos,
            eos_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0

    # Take only the continuation and optionally truncate at first EOS
    cont_ids = generated_ids[0, input_ids.shape[1]:]
    if truncate_at_eos and tokenizer.eos_token_id is not None:
        eos_positions = (cont_ids == tokenizer.eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            first_eos = int(eos_positions[0].item())
            cont_ids = cont_ids[:first_eos]

    gen_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "generated": gen_text,
        "latency_sec": round(dt, 3),
        "mask_id": mask_id,
    }


def interactive_chat(
    model_dir: str,
    max_length: int = 2048,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    avoid_eos: bool = True,
    truncate_at_eos: bool = True,
    lean_only: bool = False,
    mask_id_override: Optional[int] = None,
    system_message: Optional[str] = None,
):
    """Start an interactive chat session in the terminal.

    Maintains conversation history using the model's chat template.
    Type /exit to quit, /reset to clear history.
    """
    log(f"Loading tokenizer/model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    # Determine mask token id
    mask_id = mask_id_override
    if mask_id is None:
        mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        mask_id = 156895

    # Prepare system message
    sys_content = system_message or "You are a helpful, general-purpose AI assistant."
    if lean_only:
        sys_content += " Respond only with Lean code (import Mathlib, theorem, proof). Do not include explanations or natural language."

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": sys_content}
    ]

    print("\nInteractive chat started. Commands: /exit, /reset")
    print("Ask me anything. Press Enter to submit.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            continue
        if user_text.lower() == "/exit":
            print("Goodbye.")
            break
        if user_text.lower() == "/reset":
            messages = [{"role": "system", "content": sys_content}]
            print("History cleared.")
            continue

        messages.append({"role": "user", "content": user_text})

        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = toks["input_ids"].to(model.device)

        with torch.no_grad():
            generated_ids = llada_generate(
                model,
                input_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking='low_confidence',
                mask_id=mask_id,
                avoid_eos=avoid_eos,
                eos_token_id=tokenizer.eos_token_id,
            )

        cont_ids = generated_ids[0, input_ids.shape[1]:]
        if truncate_at_eos and tokenizer.eos_token_id is not None:
            eos_positions = (cont_ids == tokenizer.eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                first_eos = int(eos_positions[0].item())
                cont_ids = cont_ids[:first_eos]

        gen_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
        print(f"Assistant:\n{gen_text}\n")

        messages.append({"role": "assistant", "content": gen_text})


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chat with a finetuned LLaDA/LLaDA-MoE model using diffusion-style generation")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("FAST_OUTPUT_DIR", "./llada-numina-1kto21k"))
    parser.add_argument("--prompt", type=str, default=None, help="One-shot user prompt (omit or use --interactive for chat)")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--no-avoid-eos", action="store_true", help="Allow EOS during generation (discourage by default)")
    parser.add_argument("--no-truncate-at-eos", action="store_true", help="Do not cut continuation at first EOS")
    parser.add_argument("--no-lean-only", action="store_true", help="Do not force Lean-only system instruction")
    parser.add_argument("--mask-id", type=int, default=None, help="Override mask token id (default derives from config or MoE default)")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive terminal chat session")
    parser.add_argument("--system-message", type=str, default=None, help="Custom system instruction for the assistant")

    args = parser.parse_args()
    # Interactive mode takes precedence if requested or if no prompt provided
    if args.interactive or args.prompt is None:
        interactive_chat(
            model_dir=args.model_dir,
            max_length=args.max_length,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            avoid_eos=not args.no_avoid_eos,
            truncate_at_eos=not args.no_truncate_at_eos,
            lean_only=not args.no_lean_only,
            mask_id_override=args.mask_id,
            system_message=args.system_message,
        )
    else:
        result = run_chat(
            model_dir=args.model_dir,
            prompt_text=args.prompt,
            max_length=args.max_length,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            avoid_eos=not args.no_avoid_eos,
            truncate_at_eos=not args.no_truncate_at_eos,
            lean_only=not args.no_lean_only,
            mask_id_override=args.mask_id,
        )

        print("\n=== Generation Result ===")
        print(f"Latency: {result['latency_sec']}s | mask_id={result['mask_id']}")
        print("\nGenerated continuation:\n")
        print(result["generated"]) 


if __name__ == "__main__":
    main()
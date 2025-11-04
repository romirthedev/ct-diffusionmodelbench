import os
import json
import time
from datetime import datetime
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def format_instruction(example: Dict, tokenizer: AutoTokenizer, extra_system: str = "") -> Dict:
    problem = example.get('problem', '')
    formal_statement = example.get('formal_statement', '')
    formal_proof = example.get('formal_proof', '')
    answer = example.get('answer', '')

    if formal_statement:
        instruction = f"Problem: {problem}\n\nFormal Statement: {formal_statement}"
    else:
        instruction = f"Problem: {problem}"

    # For supervised inputs, the reference could be proof or short answer
    response = formal_proof if formal_proof else answer

    # Build chat-style prompt to match training
    sys_content = "You are a helpful AI assistant specialized in mathematical reasoning."
    if extra_system:
        sys_content = f"{sys_content} {extra_system}".strip()

    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    return {"prompt": prompt, "reference": response}


# ==== LLaDA-MoE diffusion-style generation utils (aligned with test_llada_moe.py) ====
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
    eos_token_id: int | None = None,
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

            # Optionally discourage EOS during generation to reduce <|endoftext|> spam
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


def build_splits_for_inference(tokenizer: AutoTokenizer,
                               training_like_max_samples: int = 100,
                               seed: int = 42,
                               split_mode: str = "test",
                               test_ratio: float = 0.10,
                               val_ratio: float = 0.10,
                               extra_system: str = ""):
    """
    Returns a dataset split according to the requested split_mode:
    - 'val_from_training': replicate the training script behavior (first N samples, then 85/15 split; returns the 15% as val)
    - 'test': create a fresh 80/10/10 split over the formatted+filtered full dataset and return the test split
    - 'val': same 80/10/10 split but return the validation split
    - 'train': return the training portion of the 80/10/10 split (for quick spot checks)
    """
    assert split_mode in {"val_from_training", "test", "val", "train"}

    if split_mode == "val_from_training":
        ds = load_dataset("AI-MO/NuminaMath-LEAN", split="train")
        ds = ds.select(range(min(training_like_max_samples, len(ds))))
        ds = ds.map(lambda x: format_instruction(x, tokenizer, extra_system=extra_system), remove_columns=ds.column_names)
        ds = ds.filter(lambda x: x["prompt"] != "")
        ds = ds.train_test_split(test_size=0.15, seed=seed)
        return ds["test"], {"mode": split_mode, "size": len(ds["test"])}, ds

    # Fresh 80/10/10 split for inference
    ds_full = load_dataset("AI-MO/NuminaMath-LEAN", split="train")
    ds_full = ds_full.map(lambda x: format_instruction(x, tokenizer, extra_system=extra_system), remove_columns=ds_full.column_names)
    ds_full = ds_full.filter(lambda x: x["prompt"] != "")

    ds_tmp = ds_full.train_test_split(test_size=test_ratio, seed=seed)
    test_ds = ds_tmp["test"]
    remain = ds_tmp["train"]
    # Normalize val portion from remaining
    remain_val_ratio = val_ratio / (1.0 - test_ratio)
    remain_split = remain.train_test_split(test_size=remain_val_ratio, seed=seed)
    train_ds, val_ds = remain_split["train"], remain_split["test"]

    if split_mode == "test":
        return test_ds, {"mode": split_mode, "size": len(test_ds)}, {"train": train_ds, "val": val_ds, "test": test_ds}
    if split_mode == "val":
        return val_ds, {"mode": split_mode, "size": len(val_ds)}, {"train": train_ds, "val": val_ds, "test": test_ds}
    if split_mode == "train":
        return train_ds, {"mode": split_mode, "size": len(train_ds)}, {"train": train_ds, "val": val_ds, "test": test_ds}


def generate_for_samples(model_dir: str,
                         split_mode: str = "test",
                         num_samples: int = 10,
                         max_length: int = 2048,
                         gen_length: int = 128,
                         steps: int = 128,
                         block_length: int = 32,
                         temperature: float = 0.0,
                         cfg_scale: float = 0.0,
                         seed: int = 42,
                         training_like_max_samples: int = 100,
                         save_dir: str = None,
                         avoid_eos: bool = True,
                         truncate_at_eos: bool = True,
                         lean_only: bool = True):
    log(f"Loading tokenizer/model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    # Build requested split
    extra_system = "Respond only with Lean code (import Mathlib, theorem, proof). Do not include explanations or natural language."
    if not lean_only:
        extra_system = ""

    split_ds, split_meta, all_splits = build_splits_for_inference(
        tokenizer,
        training_like_max_samples=training_like_max_samples,
        seed=seed,
        split_mode=split_mode,
        extra_system=extra_system,
    )

    n = min(num_samples, len(split_ds))
    log(f"Using split: {split_meta['mode']} with {split_meta['size']} examples; generating for n={n}")

    torch.manual_seed(seed)
    if hasattr(torch.cuda, "manual_seed_all"):
        torch.cuda.manual_seed_all(seed)

    results: List[Dict] = []

    for idx in range(n):
        ex = split_ds[idx]
        prompt = ex["prompt"]
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

        results.append({
            "index": idx,
            "prompt": prompt,
            "generated": gen_text,
            "reference": ex.get("reference", None),
            "latency_sec": round(dt, 3),
        })

        # Print a short preview to console
        preview = gen_text.strip().split("\n")[0][:160]
        log(f"[{idx+1}/{n}] latency={dt:.2f}s preview: {preview}")

    # Save results
    out_dir = save_dir or model_dir
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"inference_results_{split_mode}_{stamp}.jsonl")
    with open(out_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    log(f"Saved results to: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run inference of finetuned LLaDA-MoE on NuminaMath-LEAN samples")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("FAST_OUTPUT_DIR", "./Training/Training_0to1k/llada-moe-numina-finetuned-optimized"))
    parser.add_argument("--split", type=str, default="test", choices=["val_from_training", "test", "val", "train"])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-like-max-samples", type=int, default=100, help="Recreate the training split size for val_from_training mode")
    parser.add_argument("--save-dir", type=str, default=None, help="Where to save inference results (defaults to model dir)")
    parser.add_argument("--no-avoid-eos", action="store_true", help="Allow EOS during generation (by default EOS is discouraged)")
    parser.add_argument("--no-truncate-at-eos", action="store_true", help="Do not cut continuation at first EOS")
    parser.add_argument("--no-lean-only", action="store_true", help="Do not force Lean-only system instruction")

    args = parser.parse_args()

    generate_for_samples(
        model_dir=args.model_dir,
        split_mode=args.split,
        num_samples=args.num_samples,
        max_length=args.max_length,
        gen_length=args.gen_length,
        steps=args.steps,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        training_like_max_samples=args.training_like_max_samples,
        save_dir=args.save_dir,
        avoid_eos=not args.no_avoid_eos,
        truncate_at_eos=not args.no_truncate_at_eos,
        lean_only=not args.no_lean_only,
    )


if __name__ == "__main__":
    main()

import os
import json
import time
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def log(msg: str):
    print(f"[{datetime.now(). strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def add_gumbel_noise(logits: torch. Tensor, temperature: float):
    if temperature == 0:
        return logits
    logits = logits. to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits. exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index. sum(dim=1, keepdim=True)
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
    """Diffusion-style masked token generation for LLaDA(-MoE) models."""
    x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_ids.shape[1]] = prompt_ids. clone()
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
                logits = model(x). logits

            if avoid_eos and eos_token_id is not None:
                logits[..., eos_token_id] = float('-inf')

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = torch.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch. rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt_ids.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch. bool, device=x0. device)
            for j in range(confidence.shape[0]):
                _, select_index = torch. topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def load_minif2f_json(json_path: Path, split: str = "test", num_samples: Optional[int] = None) -> List[Dict]:
    """Load problems from miniF2F JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter by split
    problems = [p for p in data if p. get('split') == split]
    
    if num_samples:
        problems = problems[:num_samples]
    
    log(f"Loaded {len(problems)} problems from {split} split")
    return problems


def extract_lean_code(text: str) -> str:
    """Extract Lean code from model output, handling various formats."""
    text = text.strip()
    
    # Remove markdown code blocks if present
    if "```lean" in text:
        parts = text.split("```lean")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0]
            return code_part. strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
    
    # If no code blocks, assume entire output is code
    return text


def verify_lean4_proof(
    problem_name: str,
    header: str,
    formal_statement: str,
    generated_proof: str,
    timeout: int = 30,
    work_dir: Optional[Path] = None
) -> Tuple[bool, str]:
    """Verify a Lean 4 proof by creating a temporary file and running lean. 
    
    Returns:
        (success: bool, message: str)
    """
    # Create complete Lean file
    # The formal_statement ends with "by", so we just append the tactics
    lean_content = f"""{header}

{formal_statement}
{generated_proof}
"""
    
    # Create temporary directory for Lean project if not provided
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="lean4_verify_"))
    else:
        work_dir. mkdir(parents=True, exist_ok=True)
    
    # Create lean-toolchain file (specify Lean 4 version)
    toolchain_file = work_dir / "lean-toolchain"
    if not toolchain_file.exists():
        with open(toolchain_file, 'w') as f:
            f.write("leanprover/lean4:stable\n")
    
    # Create lakefile.lean for dependencies
    lakefile = work_dir / "lakefile.lean"
    if not lakefile.exists():
        with open(lakefile, 'w') as f:
            f. write("""import Lake
open Lake DSL

package minif2f_eval

require mathlib from git
  "https://github.com/leanprover-community/mathlib4. git"

@[default_target]
lean_lib MinifF2FEval
""")
    
    # Create the actual proof file
    proof_file = work_dir / "MinifF2FEval.lean"
    with open(proof_file, 'w') as f:
        f.write(lean_content)
    
    try:
        # First, update dependencies (only needed once per work_dir)
        if not (work_dir / "lake-packages").exists() and not (work_dir / ". lake").exists():
            log(f"Updating Lean dependencies (first time only, may take a while)...")
            update_result = subprocess.run(
                ['lake', 'update'],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes for dependency download
            )
            if update_result. returncode != 0:
                return False, f"Lake update failed: {update_result.stderr}"
        
        # Build the proof
        result = subprocess.run(
            ['lake', 'build'],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        success = result.returncode == 0
        
        if success:
            message = "Proof verified successfully"
        else:
            stderr = result.stderr
            stdout = result.stdout
            message = f"Compilation failed:\n{stderr}\n{stdout}"
        
        return success, message
    
    except subprocess.TimeoutExpired:
        return False, f"Verification timeout ({timeout}s)"
    except Exception as e:
        return False, f"Verification error: {str(e)}"


def generate_proof(
    model,
    tokenizer,
    problem: Dict,
    gen_length: int,
    steps: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    mask_id: int,
    max_length: int = 2048,
) -> str:
    """Generate a proof for a given problem. 
    
    Uses the same prompt format as chat_finetuned. py to match training.
    """
    header = problem['header']. strip()
    formal_stmt = problem['formal_statement'].strip()
    
    # Match training format: header + statement as the user content
    # The model was trained with just the Lean source, no instruction wrapping
    lean_source = f"{header}\n{formal_stmt}"
    
    # Use the same prompt format as chat_finetuned.py's build_prompt()
    messages = [
        {"role": "system", "content": "You are a helpful, general-purpose AI assistant.  Respond only with Lean code (import Mathlib, theorem, proof).  Do not include explanations or natural language. "},
        {"role": "user", "content": lean_source}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = toks["input_ids"]. to(model.device)
    
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
            avoid_eos=True,
            eos_token_id=tokenizer. eos_token_id,
        )
    
    cont_ids = generated_ids[0, input_ids.shape[1]:]
    
    # Truncate at EOS
    if tokenizer.eos_token_id is not None:
        eos_positions = (cont_ids == tokenizer.eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            first_eos = int(eos_positions[0]. item())
            cont_ids = cont_ids[:first_eos]
    
    generated_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
    
    # Extract clean Lean code if wrapped in markdown
    proof = extract_lean_code(generated_text)
    proof = proof.strip()
    
    # The formal_statement already ends with "by", so the model output
    # should be just the tactic body.  Remove leading "by" if model included it.
    if proof.lower().startswith("by"):
        proof = proof[2:]. strip()
    
    # Also handle case where model outputs ":= by"
    if proof. lower().startswith(":= by"):
        proof = proof[5:].strip()
    elif proof.lower().startswith(":="):
        proof = proof[2:].strip()
        if proof.lower().startswith("by"):
            proof = proof[2:].strip()
    
    return proof


def run_evaluation(
    model_dir: str,
    json_path: str,
    output_dir: str,
    split: str = "test",
    gen_length: int = 512,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    mask_id_override: Optional[int] = None,
    num_samples: Optional[int] = None,
    verify_proofs: bool = True,
    verification_timeout: int = 60,
    reuse_work_dir: bool = False,
):
    """Run miniF2F evaluation with Lean 4 verification."""
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log(f"Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ). eval()
    
    # Determine mask token id - same logic as chat_finetuned.py
    mask_id = mask_id_override
    if mask_id is None:
        mask_id = getattr(model. config, "mask_token_id", None)
    if mask_id is None:
        # Check tokenizer as fallback
        tokenizer_mask = getattr(tokenizer, "mask_token_id", None)
        mask_id = tokenizer_mask if tokenizer_mask is not None else 156895
    
    log(f"Using mask_id: {mask_id}")
    log(f"Loading problems from {json_path}")
    problems = load_minif2f_json(json_path, split, num_samples)
    
    # Setup work directory for Lean verification
    work_dir = None
    if verify_proofs and reuse_work_dir:
        work_dir = output_dir / "lean4_workspace"
        work_dir.mkdir(parents=True, exist_ok=True)
        log(f"Using shared Lean workspace: {work_dir}")
    
    results = []
    stats = defaultdict(int)
    
    for problem in tqdm(problems, desc="Evaluating proofs"):
        try:
            t0 = time. time()
            generated_proof = generate_proof(
                model, tokenizer, problem,
                gen_length=gen_length,
                steps=steps,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                mask_id=mask_id,
            )
            gen_time = time. time() - t0
            
            # Verify proof if requested
            verified = False
            verification_msg = "Verification skipped"
            verify_time = 0
            
            if verify_proofs:
                try:
                    t1 = time.time()
                    verified, verification_msg = verify_lean4_proof(
                        problem['name'],
                        problem['header'],
                        problem['formal_statement'],
                        generated_proof,
                        timeout=verification_timeout,
                        work_dir=work_dir
                    )
                    verify_time = time. time() - t1
                except Exception as e:
                    verification_msg = f"Verification exception: {str(e)}"
                    verify_time = 0
            
            result = {
                'name': problem['name'],
                'formal_statement': problem['formal_statement'],
                'informal_statement': problem. get('informal_statement', ''),
                'generated_proof': generated_proof,
                'verified': verified,
                'verification_message': verification_msg,
                'generation_time_sec': round(gen_time, 3),
                'verification_time_sec': round(verify_time, 3),
                'split': split
            }
            
            results.append(result)
            
            if verified:
                stats['verified'] += 1
                log(f"✓ {problem['name']} - PASS")
            else:
                log(f"✗ {problem['name']} - FAIL: {generated_proof[:100]}...")
            
            stats['total'] += 1
            
        except Exception as e:
            log(f"Error on problem {problem['name']}: {str(e)}")
            result = {
                'name': problem['name'],
                'error': str(e),
                'verified': False,
                'split': split
            }
            results.append(result)
            stats['total'] += 1
            stats['errors'] += 1
    
    # Save results
    timestamp = datetime.now(). strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"minif2f_lean4_results_{split}_{timestamp}.json"
    
    output_data = {
        'model_dir': model_dir,
        'split': split,
        'config': {
            'gen_length': gen_length,
            'steps': steps,
            'block_length': block_length,
            'temperature': temperature,
            'cfg_scale': cfg_scale,
            'mask_id': mask_id,
            'verification_timeout': verification_timeout,
        },
        'stats': {
            'total': stats['total'],
            'verified': stats['verified'],
            'errors': stats['errors'],
            'pass_rate': round(stats['verified'] / stats['total'] * 100, 2) if stats['total'] > 0 else 0.0,
        },
        'results': results,
        'timestamp': timestamp,
    }
    
    with open(results_file, 'w') as f:
        json. dump(output_data, f, indent=2)
    
    log(f"\nResults saved to {results_file}")
    log(f"\n{'='*60}")
    log(f"EVALUATION SUMMARY")
    log(f"{'='*60}")
    log(f"Total problems: {stats['total']}")
    log(f"Verified proofs: {stats['verified']}")
    log(f"Errors: {stats['errors']}")
    log(f"Pass rate: {output_data['stats']['pass_rate']:.2f}%")
    log(f"{'='*60}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaDA model on miniF2F with Lean 4 verification")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--json-path", type=str, required=True, help="Path to miniF2F JSON file")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"], help="Dataset split")
    parser.add_argument("--gen-length", type=int, default=512, help="Generation length")
    parser.add_argument("--steps", type=int, default=128, help="Diffusion steps")
    parser. add_argument("--block-length", type=int, default=32, help="Block length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--cfg-scale", type=float, default=0.0, help="CFG scale")
    parser.add_argument("--mask-id", type=int, default=None, help="Override mask token id")
    parser. add_argument("--num-samples", type=int, default=None, help="Number of problems to evaluate")
    parser.add_argument("--no-verify", action="store_true", help="Skip Lean verification")
    parser.add_argument("--verification-timeout", type=int, default=60, help="Timeout per verification (seconds)")
    parser.add_argument("--reuse-work-dir", action="store_true", help="Reuse Lean workspace (faster)")
    
    args = parser.parse_args()
    
    run_evaluation(
        model_dir=args.model_dir,
        json_path=args.json_path,
        output_dir=args. output_dir,
        split=args. split,
        gen_length=args. gen_length,
        steps=args. steps,
        block_length=args. block_length,
        temperature=args. temperature,
        cfg_scale=args. cfg_scale,
        mask_id_override=args.mask_id,
        num_samples=args.num_samples,
        verify_proofs=not args. no_verify,
        verification_timeout=args.verification_timeout,
        reuse_work_dir=args.reuse_work_dir,
    )


if __name__ == "__main__":
    main()

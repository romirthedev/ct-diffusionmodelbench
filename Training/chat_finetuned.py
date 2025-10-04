"""
LLaDA diffusion-style chat interface with progressive unmasking generation.

Key fixes vs previous version:
- Early, safe stopping: detect EOS/EOT tokens and allow them once a minimal number
  of new tokens have been written, then stop immediately.
- Contiguous growth: only unmask within a sliding frontier window so the visible
  prefix grows coherently instead of scattering far positions.
- Stabilized sampling: greedy early, then sample with a scheduled top-p; also use
  a decreasing confidence floor to avoid committing to garbage too early.
- Always pass attention_mask; no repetition penalty.

Usage:
  python llada_chat.py
"""

import os
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ==============================
# Helpers: decoding and schedules
# ==============================

def decode_contiguous_prefix(x_row: torch.Tensor, prompt_len: int, mask_id: int, tokenizer) -> str:
    """
    Decode a readable contiguous span after the prompt:
    - Stop at EOS or EOT if present after the prompt.
    - Stop at the first remaining mask after the prompt.
    """
    seq = x_row.tolist()
    end = len(seq)

    # Prefer EOT if present in additional special tokens
    stop_ids = get_stop_token_ids(tokenizer)
    # If both EOS and EOT exist, any occurrence stops decoding.
    for i in range(prompt_len, len(seq)):
        if seq[i] in stop_ids:
            end = i
            break

    for i in range(prompt_len, end):
        if seq[i] == mask_id:
            end = i
            break

    if end <= prompt_len:
        return ""
    return tokenizer.decode(
        seq[prompt_len:end],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()


def pretty_visible_prefix(x_row: torch.Tensor, prompt_len: int, mask_id: int, tokenizer) -> str:
    """
    Human-readable log view: shows the visible contiguous prefix after the prompt.
    """
    return decode_contiguous_prefix(x_row, prompt_len, mask_id, tokenizer)


def compute_tokens_to_unmask(step: int, steps: int, total_masked: int) -> int:
    """
    Linear reveal schedule: unmask a roughly equal share per remaining step.
    Ensures at least 1 token per step.
    """
    remaining_steps = max(1, steps - step)
    return max(1, math.ceil(total_masked / remaining_steps))


# ==============================
# Mask/stop resolution
# ==============================

def resolve_mask_id(model, tokenizer) -> int:
    """
    Determine the correct mask token id used for diffusion/unmasking.

    Priority:
    - model.config.mask_token_id
    - tokenizer.mask_token_id
    - tokenizer.mask_token string -> id
    - common candidates: <|mask|>, <mask>, [MASK], <MASK>
    """
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
        raise ValueError("Could not resolve a valid mask token id. Check model/tokenizer documentation.")

    # Warn if not listed as special (some tokenizers omit it, still okay)
    try:
        mask_tok = tokenizer.convert_ids_to_tokens(mask_id)
        specials = set(getattr(tokenizer, 'all_special_tokens', []))
        if mask_tok not in specials:
            print(f"[Warning] Resolved mask id {mask_id} maps to token '{mask_tok}', "
                  f"which is not listed in tokenizer.all_special_tokens.")
    except Exception:
        pass

    return mask_id


def get_stop_token_ids(tokenizer) -> set:
    """
    Collect likely stop token ids: EOS plus any end-of-turn tokens used by the chat template
    (e.g., <|im_end|>, <|eot_id|>, etc.).
    """
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    # Look for common end-of-turn tokens in special tokens/additional specials
    candidates = []
    try:
        specials_map = getattr(tokenizer, 'special_tokens_map', {}) or {}
        candidates.extend([v for k, v in specials_map.items() if isinstance(v, str) and ('eot' in k or 'eom' in k)])
    except Exception:
        pass
    try:
        addl = getattr(tokenizer, 'additional_special_tokens', []) or []
        for t in addl:
            if isinstance(t, str) and any(s in t.lower() for s in ['im_end', 'eot', 'eom', 'endofturn', 'eoa']):
                candidates.append(t)
    except Exception:
        pass

    for tok in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                stop_ids.add(tid)
        except Exception:
            continue

    return stop_ids


# ==============================
# Diffusion-style generation
# ==============================

def generate_llada(
    model,
    prompt: torch.Tensor,
    tokenizer,
    mask_id: int,
    steps: int = 80,
    gen_length: int = 256,
    temperature_start: float = 0.7,
    temperature_end: float = 0.8,
    top_p_start: float = 1.0,    # conservative early
    top_p_end: float = 0.9,      # slightly narrower near the end
    suppress_stop_until_tokens: int = 16,  # allow EOS/EOT after at least this many new tokens
    frontier_window: int = 64,   # unmask within [frontier, frontier+window)
    log_every: int = 10,
) -> str:
    """
    Progressive unmasking loop suitable for diffusion/MDM-style text models with stabilized heuristics.
    """
    print(f"[Generation Config] Steps: {steps}, Length: {gen_length}, Temp: {temperature_start}->{temperature_end}")

    device = model.device
    batch_size = prompt.shape[0]
    vocab_size = model.config.vocab_size

    # Build sequence: prompt + masked tail
    x = torch.full(
        (batch_size, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device
    )
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_len = prompt.shape[1]
    print(f"[Info] Total sequence length: {x.shape[1]} (prompt: {prompt_len}, generate: {gen_length})")

    # Attention mask: here everything is attended (no padding in this layout)
    attention_mask = torch.ones_like(x, dtype=torch.long, device=device)

    stop_ids = get_stop_token_ids(tokenizer)

    def visible_new_token_count(seq_row: torch.Tensor) -> int:
        # Count contiguous non-mask tokens after prompt until first mask or stop token
        seq = seq_row.tolist()
        n = 0
        for i in range(prompt_len, len(seq)):
            tid = seq[i]
            if tid in stop_ids or tid == mask_id:
                break
            n += 1
        return n

    for step in range(steps):
        # Early exit if no masks remain
        masked_pos = (x == mask_id).nonzero(as_tuple=False)
        if masked_pos.numel() == 0:
            print(f"[Step {step}] All tokens unmasked, stopping early")
            break

        # Early stop if a stop token appeared right after the prompt region
        new_visible = visible_new_token_count(x[0])
        if new_visible >= 1:
            # If a stop token is at the next position, cut now
            next_pos = prompt_len + new_visible
            if next_pos < x.shape[1] and x[0, next_pos] in stop_ids:
                print(f"[Step {step}] Stop token encountered; ending generation.")
                break

        # Forward pass
        with torch.no_grad():
            outputs = model(x, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits  # [B, L, V]
            logits = logits.float()  # stability for softmax on CPU/AMP

        # Schedule temperature and top-p
        progress = step / max(1, steps - 1)
        temperature = temperature_start + (temperature_end - temperature_start) * progress
        top_p = top_p_start + (top_p_end - top_p_start) * progress

        # Filtering base
        # 1) prevent generating the mask token
        logits[:, :, mask_id] = -1e9
        # 2) suppress PAD / UNK if present
        if tokenizer.pad_token_id is not None:
            logits[:, :, tokenizer.pad_token_id] = -1e9
        if tokenizer.unk_token_id is not None:
            logits[:, :, tokenizer.unk_token_id] = -1e9

        # 3) Stop tokens handling: allow early once minimal content is produced
        allow_stop = (visible_new_token_count(x[0]) >= suppress_stop_until_tokens)
        if not allow_stop:
            for sid in stop_ids:
                logits[:, :, sid] = -1e9
        else:
            # Mild bias downwards rather than ban
            for sid in stop_ids:
                logits[:, :, sid] -= 2.0

        # Compute per-position confidence
        probs_all = F.softmax(logits / max(1e-6, temperature), dim=-1)
        confidence = probs_all.max(dim=-1)[0]  # [B, L]

        # Only consider masked positions for selection
        conf_masked = confidence.clone()
        conf_masked[x != mask_id] = -1e9

        # Frontier window: only unmask near the earliest masked position to grow contiguously
        for b in range(batch_size):
            try:
                frontier = int((x[b, prompt_len:] == mask_id).nonzero(as_tuple=False)[0].item()) + prompt_len
            except Exception:
                frontier = prompt_len
            left = frontier
            right = min(x.shape[1], frontier + frontier_window)
            # Penalize positions outside the frontier window
            conf_masked[b, :left] = -1e9
            conf_masked[b, right:] = -1e9

        # Decreasing confidence floor: avoid committing at very low confidence early
        # Start higher (e.g., 0.45) and decay to 0.05 by the end
        floor_start, floor_end = 0.45, 0.05
        conf_floor = floor_start + (floor_end - floor_start) * progress

        # Determine how many to unmask this step
        total_masked_now = int((x == mask_id).sum().item())
        num_to_unmask = min(compute_tokens_to_unmask(step, steps, total_masked_now), total_masked_now)

        # Fill tokens in each batch
        for b in range(batch_size):
            # Select top confident masked positions within the frontier window
            vals, idxs = torch.topk(conf_masked[b], k=num_to_unmask)
            # Enforce confidence floor; keep at least one
            keep = idxs[vals >= conf_floor]
            if keep.numel() == 0:
                keep = idxs[:1]
            idxs = keep

            if idxs.numel() == 0:
                continue

            selected_logits = logits[b, idxs]  # [K, V]

            # Strategy: greedy early, sampling later
            if progress < 0.35:
                new_tokens = selected_logits.argmax(dim=-1)
            else:
                sel_probs = F.softmax(selected_logits / max(1e-6, temperature), dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(sel_probs, descending=True, dim=-1)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    to_remove = cum > top_p
                    # keep at least top-1
                    to_remove[:, 1:] = to_remove[:, :-1].clone()
                    to_remove[:, 0] = False
                    for i in range(sel_probs.shape[0]):
                        rem = sorted_idx[i][to_remove[i]]
                        sel_probs[i, rem] = 0.0
                    sel_probs = sel_probs / sel_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                try:
                    new_tokens = torch.multinomial(sel_probs, num_samples=1).squeeze(-1)
                except RuntimeError:
                    new_tokens = sel_probs.argmax(dim=-1)

            x[b, idxs] = torch.clamp(new_tokens, 0, vocab_size - 1)

        # Log a readable partial prefix
        if (step % log_every == 0) or (step < 3):
            prefix_text = pretty_visible_prefix(x[0], prompt_len, mask_id, tokenizer)
            print(f"[Step {step:2d}] Visible prefix: {prefix_text[-200:]}")

        # Early stop if a stop token has been inserted and visible prefix looks complete
        new_visible = visible_new_token_count(x[0])
        next_pos = prompt_len + new_visible
        if next_pos < x.shape[1] and x[0, next_pos] in stop_ids and new_visible >= 1:
            print(f"[Step {step:2d}] Stop token generated; finishing.")
            break

    # Final extraction after steps (contiguous prefix or until stop token/mask)
    answer = decode_contiguous_prefix(x[0], prompt_len, mask_id, tokenizer)

    # If still empty (e.g., masks left), replace masks with a space-like token and decode all
    if not answer:
        remaining_masks = (x == mask_id).sum().item()
        if remaining_masks > 0:
            # Try real space first
            space_id = tokenizer.convert_tokens_to_ids(' ')
            if not isinstance(space_id, int) or space_id < 0 or space_id >= vocab_size:
                try:
                    spc = tokenizer.convert_tokens_to_ids('▁')
                    if isinstance(spc, int) and 0 <= spc < vocab_size:
                        space_id = spc
                    else:
                        space_id = min(220, vocab_size - 1)  # pragmatic fallback
                except Exception:
                    space_id = min(220, vocab_size - 1)
            x[x == mask_id] = space_id
        answer = tokenizer.decode(
            x[0, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

    return answer


# ==============================
# Chat loop
# ==============================

def chat():
    """
    Interactive chat with a LLaDA-style diffusion model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using: {device}")

    # Configuration
    FINETUNED_MODEL_PATH = "./llada-numina-finetuned"  # update if needed
    USE_BASE_MODEL = False                             # set True to force base model
    USE_DIFFUSION = True                               # diffusion is required for MDM

    # Decide model path
    if USE_BASE_MODEL:
        model_path = 'GSAI-ML/LLaDA-8B-Instruct'
        print("[Model] Using base model:", model_path)
    else:
        if os.path.exists(FINETUNED_MODEL_PATH):
            model_path = FINETUNED_MODEL_PATH
            print("[Model] Using finetuned model:", model_path)
        else:
            model_path = 'GSAI-ML/LLaDA-8B-Instruct'
            print(f"[Warning] Finetuned model not found at {FINETUNED_MODEL_PATH}. Falling back to base: {model_path}")

    # Load model
    print("[Loading] Model...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if device == 'cuda' else torch.float32)
    ).to(device).eval()
    print("[Success] Model loaded")

    # Load tokenizer (prefer finetuned dir if present)
    print("[Loading] Tokenizer...")
    tokenizer_load_path = model_path if os.path.isdir(model_path) else 'GSAI-ML/LLaDA-8B-Instruct'
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_load_path,
            trust_remote_code=True,
            use_fast=False
        )
        print(f"[Success] Tokenizer loaded from: {tokenizer_load_path}")
    except Exception as e:
        print(f"[Warning] Failed to load tokenizer from {tokenizer_load_path}: {e}")
        print("[Info] Falling back to base tokenizer: GSAI-ML/LLaDA-8B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True,
            use_fast=False
        )
        print("[Success] Tokenizer loaded from base")

    # Resolve mask token
    try:
        mask_id = resolve_mask_id(model, tokenizer)
    except ValueError as e:
        print(f"[Error] {e}")
        return

    mask_tok = None
    try:
        mask_tok = tokenizer.convert_ids_to_tokens(mask_id)
    except Exception:
        pass

    # Show config basics
    print("\n" + "=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)
    print(f"Model path:            {model_path}")
    print(f"Model vocab size:      {model.config.vocab_size}")
    print(f"Tokenizer vocab size:  {tokenizer.vocab_size}")
    print(f"Mask token:            id={mask_id} token='{mask_tok}'")
    print(f"EOS token ID:          {tokenizer.eos_token_id}")
    print(f"PAD token ID:          {tokenizer.pad_token_id}")
    print(f"BOS token ID:          {tokenizer.bos_token_id}")
    print("=" * 70 + "\n")

    # System instruction to keep output in English
    system_msg = {"role": "system", "content": "You are a helpful AI assistant. Respond in English only."}
    conversation = [system_msg]

    # Generation hyperparameters
    gen_length = 256
    steps = 80
    temperature_start = 0.7
    temperature_end = 0.8
    top_p_start = 1.0
    top_p_end = 0.9
    suppress_stop_until_tokens = 16
    frontier_window = 64

    print("*" * 70)
    print(f"  Generation Mode: {'DIFFUSION' if USE_DIFFUSION else 'STANDARD'}")
    print(f"  Answer Length: {gen_length} | Steps: {steps}")
    print(f"  Temperature: {temperature_start}->{temperature_end} | Top-p: {top_p_start}->{top_p_end}")
    print(f"  Allow stop after: {suppress_stop_until_tokens} tokens | Frontier window: {frontier_window}")
    print("*" * 70)
    print("Type 'quit' to exit\n")

    # Chat loop
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

        conversation.append({"role": "user", "content": user_input})

        # Apply chat template
        try:
            prompt_text = tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            prompt_text = f"System: Respond in English only.\nUser: {user_input}\nAssistant:"

        # Tokenize
        input_ids = tokenizer(prompt_text, return_tensors="pt")['input_ids'].to(device)
        print(f"\n[Generating] Input length: {input_ids.shape[1]} tokens")

        # Generate
        if not USE_DIFFUSION:
            # Not recommended for diffusion models; provided for completeness
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=gen_length,
                    temperature=temperature_end,
                    do_sample=True,
                    top_p=top_p_end,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
            answer = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        else:
            answer = generate_llada(
                model=model,
                prompt=input_ids,
                tokenizer=tokenizer,
                mask_id=mask_id,
                steps=steps,
                gen_length=gen_length,
                temperature_start=temperature_start,
                temperature_end=temperature_end,
                top_p_start=top_p_start,
                top_p_end=top_p_end,
                suppress_stop_until_tokens=suppress_stop_until_tokens,
                frontier_window=frontier_window,
                log_every=10,
            )

        print("\n" + "=" * 70)
        print("Assistant:", answer)
        print("=" * 70 + "\n")

        conversation.append({"role": "assistant", "content": answer})
        # Keep context manageable
        if len(conversation) > 20:
            conversation = [system_msg] + conversation[-19:]


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" LLaDA Diffusion Chat Interface ")
    print("=" * 70 + "\n")
    try:
        chat()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        import traceback
        traceback.print_exc()
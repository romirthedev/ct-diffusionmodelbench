"""
LLaDA diffusion-style chat interface with progressive unmasking generation.

Stability upgrades:
- Sequential frontier fill (1–2 tokens/step) with re-forward per token.
- Stronger stop handling:
  * Allow stop tokens after a small minimum.
  * If stop prob is high and we have a full sentence, force-stop.
  * Optional max_sentences early stop heuristic.
- Repetition controls:
  * Immediate unigram block
  * Presence/frequency penalties over a recent window
  * No-repeat 3-gram blocking
- Punctuation run limiter:
  * Detect punctuation bursts, cool down sampling, and penalize punct tokens.

Usage:
  python llada_chat.py
"""

import os
import re
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

    stop_ids = get_stop_token_ids(tokenizer)
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
    return decode_contiguous_prefix(x_row, prompt_len, mask_id, tokenizer)


def compute_tokens_to_unmask(step: int, steps: int, total_masked: int, max_per_step: int = 2) -> int:
    """
    Keep per-step reveals small to avoid simultaneous identical picks.
    """
    return max(1, min(max_per_step, total_masked // max(1, (steps - step)) + 1))


def sentence_count(text: str) -> int:
    """
    Rough sentence terminator count; '...' counts as one.
    """
    # Collapse ellipses first
    t = re.sub(r"\.\.+", ".", text)
    return len(re.findall(r"[.!?]", t))


# ==============================
# Mask/stop resolution
# ==============================

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
        raise ValueError("Could not resolve a valid mask token id. Check model/tokenizer documentation.")

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
    Collect likely stop token ids: EOS plus any end-of-turn tokens, including common aliases.
    """
    stop_ids = set()
    # Known attributes that some tokenizers expose
    for attr in ["eos_token_id", "eot_token_id", "im_end_id"]:
        tid = getattr(tokenizer, attr, None)
        if isinstance(tid, int) and tid >= 0:
            stop_ids.add(int(tid))

    # Common special token strings
    candidates = []
    try:
        specials_map = getattr(tokenizer, 'special_tokens_map', {}) or {}
        for k, v in specials_map.items():
            if isinstance(v, str) and any(s in k.lower() for s in ['eot', 'eom', 'im_end']):
                candidates.append(v)
    except Exception:
        pass
    try:
        addl = getattr(tokenizer, 'additional_special_tokens', []) or []
        for t in addl:
            if isinstance(t, str) and any(s in t.lower() for s in ['im_end', 'eot', 'eom', 'endofturn', 'eoa']):
                candidates.append(t)
    except Exception:
        pass
    # Hard-coded aliases often used
    candidates.extend(['<|eot_id|>', '<|im_end|>'])

    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    for tok in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                stop_ids.add(tid)
        except Exception:
            continue

    return stop_ids


# ==============================
# Repetition controls
# ==============================

def apply_presence_frequency_penalties(logits_row: torch.Tensor,
                                       context: torch.Tensor,
                                       presence_penalty: float = 0.45,
                                       frequency_penalty: float = 0.2,
                                       window: int = 128):
    """
    OpenAI-style presence/frequency penalties on a recent context window.
    """
    if window > 0:
        ctx = context[-window:].tolist()
    else:
        ctx = context.tolist()
    if not ctx:
        return
    from collections import Counter
    c = Counter(ctx)
    for tok_id, cnt in c.items():
        if 0 <= tok_id < logits_row.shape[-1]:
            logits_row[tok_id] -= (presence_penalty + frequency_penalty * cnt)


def block_immediate_repeat(logits_row: torch.Tensor, last_token_id: int):
    """
    Disallow repeating the immediately previous token.
    """
    if last_token_id is not None and 0 <= last_token_id < logits_row.shape[-1]:
        logits_row[last_token_id] = -1e9


def build_no_repeat_ngram_map(tokens: list[int], n: int) -> dict:
    """
    Map: (n-1)-gram tuple -> set of disallowed next tokens that would recreate a seen n-gram.
    """
    from collections import defaultdict
    banned = defaultdict(set)
    if n <= 1 or len(tokens) < n:
        return banned
    for i in range(len(tokens) - n + 1):
        prefix = tuple(tokens[i:i + n - 1])
        nxt = tokens[i + n - 1]
        banned[prefix].add(nxt)
    return banned


def apply_no_repeat_ngram_block(logits_row: torch.Tensor, history_tokens: list[int], n: int = 3):
    """
    Apply no-repeat n-gram blocking for n=3 by default.
    """
    if n <= 1 or len(history_tokens) < n - 1:
        return
    banned_map = build_no_repeat_ngram_map(history_tokens, n)
    prefix = tuple(history_tokens[-(n - 1):])
    banned = banned_map.get(prefix, ())
    for tok in banned:
        if 0 <= tok < logits_row.shape[-1]:
            logits_row[tok] = -1e9


def get_punctuation_token_ids(tokenizer) -> set:
    """
    Best-effort set of punctuation tokens across common tokenizers (SP/BBPE/BPE).
    """
    punct_char_seeds = ['.', '!', '?', ',', ';', ':', '…']
    prefixes = ['', ' ', '  ', '▁', 'Ġ']  # SP and BBPE/ GPT2 style whitespace markers
    ids = set()
    for p in prefixes:
        for ch in punct_char_seeds:
            for s in [f"{p}{ch}", f"{p}{ch}{ch}"]:  # single and doubled (.., !!)
                try:
                    tid = tokenizer.convert_tokens_to_ids(s)
                    if isinstance(tid, int) and tid >= 0:
                        ids.add(tid)
                except Exception:
                    pass
    return ids


def punct_run_detect(tokenizer, left_ctx_ids: torch.Tensor, k_chars: int = 16, threshold: float = 0.6) -> bool:
    """
    Detect if the last k_chars are dominated by punctuation; helps damp punctuation cascades.
    """
    if left_ctx_ids.numel() == 0:
        return False
    # Decode a small suffix; if this is slow for your tokenizer, raise the slice limit.
    tail_text = tokenizer.decode(left_ctx_ids[-64:].tolist(), skip_special_tokens=True)
    if not tail_text:
        return False
    tail = tail_text[-k_chars:]
    punct_chars = set('.!,?;:…·•-–—()[]{}\'"')
    if len(tail) == 0:
        return False
    ratio = sum(ch in punct_chars for ch in tail) / len(tail)
    return ratio >= threshold


# ==============================
# Diffusion-style generation
# ==============================

def generate_llada(
    model,
    prompt: torch.Tensor,
    tokenizer,
    mask_id: int,
    steps: int = 80,
    gen_length: int = 192,
    temperature_start: float = 0.65,
    temperature_end: float = 0.75,
    top_p_start: float = 0.95,
    top_p_end: float = 0.75,
    suppress_stop_until_tokens: int = 8,
    tokens_per_step: int = 2,
    no_repeat_ngram_size: int = 3,
    p_stop_force_threshold: float = 0.40,  # if stop prob exceeds this after a sentence, force stop
    max_sentences: int | None = None,      # set to 1 or 2 to stop after N sentences
    log_every: int = 10,
) -> str:
    """
    Progressive unmasking loop with sequential frontier filling and robust tail controls.
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

    attention_mask = torch.ones_like(x, dtype=torch.long, device=device)
    stop_ids = get_stop_token_ids(tokenizer)
    punct_ids = get_punctuation_token_ids(tokenizer)

    def visible_new_token_count(seq_row: torch.Tensor) -> int:
        seq = seq_row.tolist()
        n = 0
        for i in range(prompt_len, len(seq)):
            tid = seq[i]
            if tid in stop_ids or tid == mask_id:
                break
            n += 1
        return n

    def earliest_mask_pos(seq_row: torch.Tensor) -> int:
        pos = (seq_row[prompt_len:] == mask_id).nonzero(as_tuple=False)
        if pos.numel() == 0:
            return -1
        return int(pos[0].item()) + prompt_len

    for step in range(steps):
        # Early exit if no masks remain
        if (x == mask_id).sum().item() == 0:
            print(f"[Step {step}] All tokens unmasked, stopping early")
            break

        # Early stop if a stop token appears immediately after visible prefix
        new_visible = visible_new_token_count(x[0])
        if new_visible >= 1:
            next_pos = prompt_len + new_visible
            if next_pos < x.shape[1] and int(x[0, next_pos].item()) in stop_ids:
                print(f"[Step {step}] Stop token encountered; ending generation.")
                break

        # Schedule temperature and top-p
        progress = step / max(1, steps - 1)
        base_temperature = temperature_start + (temperature_end - temperature_start) * progress
        base_top_p = top_p_start + (top_p_end - top_p_start) * progress

        # Decide how many tokens to place this step (keep small)
        total_masked_now = int((x == mask_id).sum().item())
        k_this = min(compute_tokens_to_unmask(step, steps, total_masked_now, max_per_step=tokens_per_step),
                     total_masked_now)

        # For each batch, fill sequentially from the frontier and recompute logits each time
        for b in range(batch_size):
            for k in range(k_this):
                pos = earliest_mask_pos(x[b])
                if pos < 0:
                    break

                # Forward pass on current canvas
                with torch.no_grad():
                    outputs = model(x, attention_mask=attention_mask, use_cache=False)
                    logits = outputs.logits.float()  # [B, L, V]

                logits_row = logits[b, pos, :]

                # Base filtering
                logits_row[mask_id] = -1e9
                if tokenizer.pad_token_id is not None:
                    logits_row[tokenizer.pad_token_id] = -1e9
                if tokenizer.unk_token_id is not None and 0 <= tokenizer.unk_token_id < logits_row.shape[-1]:
                    logits_row[tokenizer.unk_token_id] = -1e9

                # Stop tokens handling
                allow_stop = (visible_new_token_count(x[b]) >= suppress_stop_until_tokens)
                # We will compute softmax later; for now, optionally ban or bias stop tokens
                if not allow_stop:
                    for sid in stop_ids:
                        if 0 <= sid < logits_row.shape[-1]:
                            logits_row[sid] = -1e9
                else:
                    for sid in stop_ids:
                        if 0 <= sid < logits_row.shape[-1]:
                            logits_row[sid] -= 1.0  # mild bias down, but not a hard ban

                # Repetition controls
                left_ctx = x[b, :pos]
                last_token_id = int(left_ctx[-1].item()) if left_ctx.numel() > 0 else None
                block_immediate_repeat(logits_row, last_token_id)
                apply_presence_frequency_penalties(
                    logits_row,
                    left_ctx,
                    presence_penalty=0.4,
                    frequency_penalty=0.15,
                    window=128
                )
                # No-repeat tri-gram
                apply_no_repeat_ngram_block(logits_row, left_ctx.tolist(), n=no_repeat_ngram_size)

                # Punctuation run limiter
                temperature = base_temperature
                top_p = base_top_p
                if punct_run_detect(tokenizer, left_ctx, k_chars=16, threshold=0.55):
                    # Cool down and penalize punctuation further
                    temperature = min(temperature, 0.55)
                    top_p = min(top_p, 0.55)
                    for pid in punct_ids:
                        if 0 <= pid < logits_row.shape[-1]:
                            logits_row[pid] -= 1.5

                # Compute probabilities for stop forcing logic
                probs_pre = F.softmax(logits_row / max(1e-6, temperature), dim=-1)

                # If allowed to stop and we have at least one sentence, consider forcing stop
                if allow_stop:
                    prefix_text = pretty_visible_prefix(x[b], prompt_len, mask_id, tokenizer)
                    if sentence_count(prefix_text) >= 1:
                        p_stop = float(sum(probs_pre[sid].item() for sid in stop_ids if 0 <= sid < probs_pre.shape[-1]))
                        if p_stop >= p_stop_force_threshold:
                            # Choose the most likely stop token and finish
                            best_sid = max(stop_ids, key=lambda sid: probs_pre[sid].item() if 0 <= sid < probs_pre.shape[-1] else -1.0)
                            best_sid = best_sid if 0 <= best_sid < logits_row.shape[-1] else tokenizer.eos_token_id
                            x[b, pos] = int(best_sid)
                            print(f"[Step {step:2d}] Forced stop at pos {pos} (p_stop={p_stop:.2f}).")
                            break

                # Sample actual token
                if progress < 0.30:
                    # Greedy early
                    new_tok = int(torch.argmax(logits_row).item())
                else:
                    probs = F.softmax(logits_row / max(1e-6, temperature), dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                        cum = torch.cumsum(sorted_probs, dim=-1)
                        to_remove = cum > top_p
                        # keep at least top-1
                        to_remove[1:] = to_remove[:-1].clone()
                        to_remove[0] = False
                        probs = probs.clone()
                        probs[sorted_idx[to_remove]] = 0.0
                        probs = probs / probs.sum().clamp_min(1e-12)
                    try:
                        new_tok = int(torch.multinomial(probs, num_samples=1).item())
                    except RuntimeError:
                        new_tok = int(torch.argmax(probs).item())

                x[b, pos] = max(0, min(vocab_size - 1, new_tok))

                # If we just generated a stop token, stop now
                if allow_stop and new_tok in stop_ids and visible_new_token_count(x[b]) >= 1:
                    print(f"[Step {step:2d}] Stop token generated at pos {pos}; finishing.")
                    break

                # Optional: stop after N sentences for safety
                if max_sentences is not None:
                    prefix_text = pretty_visible_prefix(x[b], prompt_len, mask_id, tokenizer)
                    if sentence_count(prefix_text) >= max_sentences and visible_new_token_count(x[b]) >= suppress_stop_until_tokens:
                        print(f"[Step {step:2d}] Reached max_sentences={max_sentences}; finishing.")
                        break

        # Log
        if (step % log_every == 0) or (step < 3):
            prefix_text = pretty_visible_prefix(x[0], prompt_len, mask_id, tokenizer)
            print(f"[Step {step:2d}] Visible prefix: {prefix_text[-200:]}")

        # Early stop if stop was placed right after the visible prefix
        new_visible = visible_new_token_count(x[0])
        next_pos = prompt_len + new_visible
        if next_pos < x.shape[1] and int(x[0, next_pos].item()) in stop_ids and new_visible >= 1:
            print(f"[Step {step:2d}] Stop token at frontier; finishing.")
            break

    # Final extraction
    answer = decode_contiguous_prefix(x[0], prompt_len, mask_id, tokenizer)

    if not answer:
        remaining_masks = (x == mask_id).sum().item()
        if remaining_masks > 0:
            # Replace any remaining masks with a whitespace-like token and decode
            space_id = tokenizer.convert_tokens_to_ids(' ')
            if not isinstance(space_id, int) or space_id < 0 or space_id >= vocab_size:
                try:
                    spc = tokenizer.convert_tokens_to_ids('▁')
                    if isinstance(spc, int) and 0 <= spc < vocab_size:
                        space_id = spc
                    else:
                        space_id = min(220, vocab_size - 1)
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
    FINETUNED_MODEL_PATH = "./llada-numina-finetuned"
    USE_BASE_MODEL = False
    USE_DIFFUSION = True

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

    # Load tokenizer
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

    # Generation hyperparameters (safer defaults)
    gen_length = 192
    steps 1024 # from 80
    temperature_start = 0.65
    temperature_end = 0.75
    top_p_start = 0.95
    top_p_end = 0.75
    suppress_stop_until_tokens = 8
    tokens_per_step = 2
    no_repeat_ngram_size = 3
    p_stop_force_threshold = 0.40
    max_sentences = 2  # Stop after two sentences by default

    print("*" * 70)
    print(f"  Generation Mode: {'DIFFUSION' if USE_DIFFUSION else 'STANDARD'}")
    print(f"  Answer Length: {gen_length} | Steps: {steps}")
    print(f"  Temperature: {temperature_start}->{temperature_end} | Top-p: {top_p_start}->{top_p_end}")
    print(f"  Allow stop after: {suppress_stop_until_tokens} tokens | Tokens/step: {tokens_per_step}")
    print(f"  No-repeat n-gram size: {no_repeat_ngram_size} | Max sentences: {max_sentences}")
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
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=gen_length,
                    temperature=temperature_end,
                    do_sample=True,
                    top_p=top_p_end,
                    repetition_penalty=1.1,
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
                tokens_per_step=tokens_per_step,
                no_repeat_ngram_size=no_repeat_ngram_size,
                p_stop_force_threshold=p_stop_force_threshold,
                max_sentences=max_sentences,
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
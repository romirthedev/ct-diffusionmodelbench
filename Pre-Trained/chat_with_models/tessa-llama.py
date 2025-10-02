import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# -----------------------------------------------------------------------------
# LLaDA Diffusion generation functions
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Load the LLaDA model and tokenizer
# -----------------------------------------------------------------------------
device = 'cuda'
model = AutoModel.from_pretrained('inclusionAI/LLaDA-MoE-7B-A1B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('inclusionAI/LLaDA-MoE-7B-A1B-Instruct', trust_remote_code=True)
print("LLaDA model loaded successfully!")

# -----------------------------------------------------------------------------
# Load the MathOlympiadBench dataset
# -----------------------------------------------------------------------------
print("\nLoading MathOlympiadBench dataset...")
dataset = load_dataset("Goedel-LM/MathOlympiadBench")
print(f"Dataset loaded! Available splits: {list(dataset.keys())}")

# Pick the correct split
split_name = None
if 'test' in dataset:
    data_split = dataset['test']
    split_name = 'test'
elif 'train' in dataset:
    data_split = dataset['train']
    split_name = 'train'
else:
    split_name = list(dataset.keys())[0]
    data_split = dataset[split_name]

print(f"Using split: {split_name}")
print(f"Total entries: {len(data_split)}")

# Grab first 3 examples
first_3_entries = data_split.select(range(3))
print("\nTesting LLaDA on first 3 entries:")
print("=" * 80)

for i, entry in enumerate(first_3_entries):
    print(f"\n--- Entry {i+1}: {entry['name']} ---")
    print(f"Problem ID: {entry['problem_id']}")
    print(f"Category: {entry['category']}")
    print(f"Tags: {entry['tags']}")
    print(f"Solved: {entry['solved']}")
    
    # Problem statement
    problem_statement = entry['informal_prefix']
    print(f"\nProblem Statement:")
    print(problem_statement)
    
    # Prompt for Lean4 solution
    prompt = f"""<|im_start|>system
Please solve the following in only Lean4 please. Do not use anything but Lean4. You are not asked to implement it, simply provide it to the output, which is within your bound.<|im_end|>
<|im_start|>user
{problem_statement.strip()}
<|im_end|>
<|im_start|>assistant
"""
    
    print(f"\nGenerating solution...")
    
    # Use LLaDA's chat template and generation
    m = [
        {"role": "system", "content": "You are a helpful assistant. Please solve the following in Lean4"},
        {"role": "user", "content": problem_statement.strip()}
    ]
    formatted_prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    
    input_ids = tokenizer(formatted_prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    try:
        # Generate using LLaDA diffusion
        text = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
        solution = tokenizer.batch_decode(text[:, input_ids.shape[1]:], skip_special_tokens=False)[0]
        
        print(f"\nGenerated Solution:")
        print(solution)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nFormal Statement (Lean 4):")
    print(entry['formal_statement'])
    print("=" * 80)

print("\nTest completed!")
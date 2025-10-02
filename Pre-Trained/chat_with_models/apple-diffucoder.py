import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

# Load the model and tokenizer
model_path = "apple/DiffuCoder-7B-cpGRPO"
print(f"Loading model: {model_path}")
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to("cuda").eval()
print("Model loaded successfully!")

# Load the MathOlympiadBench dataset
print("\nLoading MathOlympiadBench dataset...")
dataset = load_dataset("Goedel-LM/MathOlympiadBench")
print(f"Dataset loaded! Available splits: {list(dataset.keys())}")

# The dataset likely has a different split name, let's check
if 'test' in dataset:
    data_split = dataset['test']
elif 'train' in dataset:
    data_split = dataset['train'] 
else:
    # Get the first available split
    split_name = list(dataset.keys())[0]
    data_split = dataset[split_name]
    print(f"Using split: {split_name}")

print(f"Total entries: {len(data_split)}")

# Get the first 3 entries
first_3_entries = data_split.select(range(3))

print("\nTesting DiffuCoder on first 3 entries:")
print("=" * 80)

TOKEN_PER_STEP = 1  # diffusion timesteps * TOKEN_PER_STEP = total new tokens

for i, entry in enumerate(first_3_entries):
    print(f"\n--- Entry {i+1}: {entry['name']} ---")
    print(f"Problem ID: {entry['problem_id']}")
    print(f"Category: {entry['category']}")
    print(f"Tags: {entry['tags']}")
    print(f"Solved: {entry['solved']}")
    
    # Extract the problem statement from informal_prefix
    problem_statement = entry['informal_prefix']
    print(f"\nProblem Statement:")
    print(problem_statement)
    
    # Create the prompt using the problem statement
    prompt = f"""<|im_start|>system
You are a helpful assistant. Please solve the following in Lean4<|im_end|>
<|im_start|>user
{problem_statement.strip()}
<|im_end|>
<|im_start|>assistant
"""
    
    print(f"\nGenerating solution...")
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device="cuda")
    attention_mask = inputs.attention_mask.to(device="cuda")
    
    try:
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            output_history=True,
            return_dict_in_generate=True,
            steps=256//TOKEN_PER_STEP,
            temperature=0.4,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.,
        )
        
        generations = [
            tokenizer.decode(g[len(p):].tolist())
            for p, g in zip(input_ids, output.sequences)
        ]
        
        solution = generations[0].split('<|dlm_pad|>')[0]
        print(f"\nGenerated Solution:")
        print(solution)
        
    except Exception as e:
        print(f"Error during generation: {e}")
    
    print(f"\nFormal Statement (Lean 4):")
    print(entry['formal_statement'])
    
    print("=" * 80)

print("\nTest completed!")
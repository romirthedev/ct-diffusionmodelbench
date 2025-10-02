import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer; avoid custom classes missing in repo."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print("AutoModelForCausalLM loaded successfully")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def generate_response(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.7):
    """Generate response, returns the output after the prompt."""
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    # If decoded starts with the prompt, strip it off
    if decoded.startswith(prompt):
        return decoded[len(prompt):].strip()
    else:
        return decoded.strip()

def create_simple_prompt(informal_description, formal_statement):
    """Simple prompt form: ask to replace 'sorry' with proof."""
    return f"""Prove this theorem in Lean 4:

Problem: {informal_description}

Formal statement:
{formal_statement}

Replace 'sorry' with the complete proof:
"""

def main():
    model_name = "maple-research-lab/LLaDOU-v0-Math"
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)
    if model is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        return

    print("Model and tokenizer ready.")
    print(f"Device: {next(model.parameters()).device}")

    problems = [
        {
            "informal": "Prove that there do not exist integers x,y,z such that x⁶ + x³ + x³y + y = 147¹⁵⁷ and x³ + x³y + y² + y + z⁹ = 157¹⁴⁷.",
            "formal": "theorem usa2005_p2 : ¬∃ (x y z : ℤ), x^6 + x^3 + x^3 * y + y = 147^157 ∧ x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := by sorry",
        },
        {
            "informal": "Determine the least real number M such that |ab(a² - b²) + bc(b² - c²) + ca(c² - a²)| ≤ M (a² + b² + c²)² for all real numbers a,b,c.",
            "formal": """noncomputable abbrev solution : ℝ := 9 * Real.sqrt 2 / 32
theorem imo2006_p3 : IsLeast { M | (∀ a b c : ℝ, |a * b * (a ^ 2 - b ^ 2) + b * c * (b ^ 2 - c ^ 2) + c * a * (c ^ 2 - a ^ 2)| ≤ M * (a ^ 2 + b ^ 2 + c ^ 2) ^ 2) } solution := by sorry""",
        },
        # Add more problems as needed
    ]

    output_dir = "lean_outputs"
    os.makedirs(output_dir, exist_ok=True)

    for i, prob in enumerate(problems, start=1):
        print(f"\n=== Problem {i} ===")
        print("Informal:", prob["informal"])
        prompt = create_simple_prompt(prob["informal"], prob["formal"])
        print(f"Prompt length: {len(prompt)} characters")

        try:
            proof = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=2048,
                temperature=0.7
            )
            print("Proof generated.")

            lean_code = proof  # you might want to prepend the statement etc.

            # Save to a .lean file
            filename = os.path.join(output_dir, f"problem_{i}.lean")
            with open(filename, "w") as f:
                f.write(lean_code)

            print(f"Saved Lean proof to {filename}")
            print("Proof:")
            print(lean_code)

        except Exception as e:
            print(f"Error generating/saving for problem {i}: {e}")

    print("\nAll done.")

if __name__ == "__main__":
    main()

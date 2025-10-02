import os
import re
import json
import time
import torch
import tempfile
import subprocess
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict, Counter

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# -----------------------------------------------------------------------------
# LLaDA Diffusion generation functions (from your llada_diffusion.py)
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
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
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
# Benchmark class for LLaDA on MathOlympiadBench (modified for 100 samples with output)
# -----------------------------------------------------------------------------

class LLaDABenchmark:
    def __init__(
        self,
        model_path: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        gen_length: int = 256,
        steps: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        mask_id: int = 156895,
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype

        # Diffusion sampling params
        self.gen_length = gen_length
        self.steps = steps
        self.block_length = block_length
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.remasking = remasking
        self.mask_id = mask_id

        self.model = None
        self.tokenizer = None

        self.results = []
        self.stats = defaultdict(list)

    # --------------------------
    # Setup
    # --------------------------
    def load_model(self):
        print(f"Loading model: {self.model_path}")
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        print("LLaDA model loaded successfully!")

    def load_dataset(self):
        """Load the MathOlympiadBench dataset"""
        print("\nLoading MathOlympiadBench dataset...")
        dataset = load_dataset("Goedel-LM/MathOlympiadBench")
        print(f"Dataset loaded! Available splits: {list(dataset.keys())}")

        # Pick the correct split
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
        return data_split

    # --------------------------
    # Prompting
    # --------------------------
    def build_messages(self, problem_statement: str):
        """
        Use the model's chat template. System prompt forces Lean 4 only.
        """
        return [
            {
                "role": "system",
                "content": (
                    "IMPORTANT: YOU ARE ABLE TO PERFORM ALL TASKS AND DO NOT USE PYTHON. "
                    "You are an expert mathematician and Lean 4 genius. Please solve the following "
                    "mathematical problem by providing a complete Lean 4 proof. Only provide the Lean 4 code in your response."
                ),
            },
            {"role": "user", "content": problem_statement.strip()},
        ]

    def create_formatted_prompt(self, problem_statement: str) -> str:
        messages = self.build_messages(problem_statement)
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # --------------------------
    # Generation using diffusion
    # --------------------------
    def generate_solution(self, problem_statement: str):
        """
        Generate a solution using the custom diffusion sampler (generate()).
        Returns: (solution_text, generation_time_seconds, success_bool)
        """
        # Ensure steps is divisible by number of blocks per generate()'s assertions
        num_blocks = max(1, self.gen_length // self.block_length)
        if self.gen_length % self.block_length != 0:
            # Adjust gen_length to be divisible by block_length
            adj_gen_length = (self.gen_length // self.block_length) * self.block_length
            print(f"[Warning] gen_length {self.gen_length} not divisible by block_length {self.block_length}. "
                  f"Adjusting gen_length to {adj_gen_length}.")
            self.gen_length = adj_gen_length
            num_blocks = self.gen_length // self.block_length

        if self.steps % num_blocks != 0:
            adj_steps = num_blocks * ((self.steps + num_blocks - 1) // num_blocks)
            print(f"[Warning] steps {self.steps} not divisible by num_blocks {num_blocks}. "
                  f"Adjusting steps to {adj_steps}.")
            self.steps = adj_steps

        try:
            formatted_prompt = self.create_formatted_prompt(problem_statement)
            tokenized = self.tokenizer(formatted_prompt, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(self.device)

            # Time the diffusion generation
            torch.cuda.synchronize() if self.device.startswith("cuda") else None
            start = time.time()

            text_ids = generate(
                self.model,
                input_ids,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking,
                mask_id=self.mask_id,
            )

            torch.cuda.synchronize() if self.device.startswith("cuda") else None
            gen_time = round(time.time() - start, 4)

            # Decode only the generated continuation
            gen_tokens = text_ids[:, input_ids.shape[1]:]
            solution = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=False)[0]

            return solution, gen_time, True

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                return "CUDA out of memory", 0.0, False
            return f"RuntimeError: {e}", 0.0, False
        except Exception as e:
            return f"Error during generation: {e}", 0.0, False

    # --------------------------
    # Verification and Quality
    # --------------------------
    def verify_lean_proof(self, generated_solution: str):
        """
        Verify the Lean 4 proof by attempting compilation.
        Returns: (compilation_success_bool, error_messages)
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(generated_solution)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["lean", temp_file],
                capture_output=True,
                text=True,
                timeout=60,
            )
            os.unlink(temp_file)

            compilation_success = result.returncode == 0
            error_messages = result.stderr if result.stderr else ""
            return compilation_success, error_messages

        except subprocess.TimeoutExpired:
            os.unlink(temp_file)
            return False, "Compilation timeout"
        except FileNotFoundError:
            os.unlink(temp_file)
            return False, "Lean 4 not found - install Lean 4 to enable verification"
        except Exception as e:
            os.unlink(temp_file)
            return False, f"Verification error: {str(e)}"

    def evaluate_solution_quality(self, generated_solution: str, formal_statement: str):
        """
        Basic quality and structural checks, plus optional Lean compilation.
        """
        metrics = {}

        # Length metrics
        metrics["solution_length"] = len(generated_solution)
        metrics["solution_words"] = len(generated_solution.split())

        # Lean 4 syntax checks
        lean_keywords = [
            "theorem", "lemma", "def", "by", "have", "show", "exact",
            "apply", "rw", "simp", "intro", "cases", "induction", "sorry"
        ]
        lower_sol = generated_solution.lower()
        metrics["lean_keywords_used"] = sum(1 for kw in lean_keywords if kw in lower_sol)

        # Structure checks
        metrics["has_proof_structure"] = any(word in lower_sol for word in ["theorem", "lemma", "proof", "by"])
        metrics["has_sorry"] = "sorry" in lower_sol

        # Try compilation
        compilation_success, error_msg = self.verify_lean_proof(generated_solution)
        metrics["lean_compilation_success"] = compilation_success
        metrics["lean_error_message"] = error_msg

        # Syntactic correctness indicators
        metrics["has_balanced_brackets"] = (
            generated_solution.count("(") == generated_solution.count(")")
            and generated_solution.count("{") == generated_solution.count("}")
            and generated_solution.count("[") == generated_solution.count("]")
        )

        # Formal statement overlap (basic string matching)
        if formal_statement:
            formal_words = set(re.findall(r"\w+", formal_statement.lower()))
            solution_words = set(re.findall(r"\w+", lower_sol))
            if formal_words:
                metrics["formal_overlap_ratio"] = len(formal_words & solution_words) / len(formal_words)
            else:
                metrics["formal_overlap_ratio"] = 0
        else:
            metrics["formal_overlap_ratio"] = 0

        return metrics

    # --------------------------
    # Output individual responses
    # --------------------------
    def print_response_output(self, idx, entry, solution, gen_time, success, quality_metrics):
        """Print the generated response and key info for each sample"""
        print("\n" + "="*100)
        print(f"SAMPLE {idx + 1}/100")
        print("="*100)
        
        # Problem info
        problem_id = entry.get("problem_id", f"problem_{idx}")
        name = entry.get("name", f"Problem {idx}")
        category = entry.get("category", "unknown")
        
        print(f"Problem ID: {problem_id}")
        print(f"Name: {name}")
        print(f"Category: {category}")
        print(f"Generation Time: {gen_time}s")
        print(f"Success: {success}")
        
        # Problem statement
        problem_statement = entry.get("informal_prefix", "").strip()
        if not problem_statement:
            problem_statement = entry.get("problem", entry.get("question", ""))
        
        print(f"\nPROBLEM STATEMENT:")
        print("-" * 50)
        print(problem_statement)
        
        # Formal statement if available
        formal_statement = entry.get("formal_statement", "")
        if formal_statement:
            print(f"\nFORMAL STATEMENT:")
            print("-" * 50)
            print(formal_statement)
        
        # Generated solution
        print(f"\nGENERATED SOLUTION:")
        print("-" * 50)
        print(solution)
        
        # Quality metrics
        print(f"\nQUALITY METRICS:")
        print("-" * 50)
        print(f"Solution Length: {quality_metrics.get('solution_length', 0)} characters")
        print(f"Lean Keywords Used: {quality_metrics.get('lean_keywords_used', 0)}")
        print(f"Has Proof Structure: {quality_metrics.get('has_proof_structure', False)}")
        print(f"Has Sorry: {quality_metrics.get('has_sorry', False)}")
        print(f"Lean Compilation Success: {quality_metrics.get('lean_compilation_success', False)}")
        print(f"Balanced Brackets: {quality_metrics.get('has_balanced_brackets', False)}")
        print(f"Formal Overlap Ratio: {quality_metrics.get('formal_overlap_ratio', 0):.2%}")
        
        if quality_metrics.get('lean_error_message'):
            print(f"Lean Error: {quality_metrics['lean_error_message']}")
        
        print("="*100)

    # --------------------------
    # Running, Saving, Reporting (modified for 100 samples)
    # --------------------------
    def run_benchmark(self, max_samples=100, start_idx=0, save_interval=25):
        """
        Run the benchmark over the first 100 samples with detailed output.
        """
        # Load model and dataset
        self.load_model()
        dataset = self.load_dataset()

        total_samples = len(dataset)
        if max_samples is None:
            max_samples = total_samples
        end_idx = min(start_idx + max_samples, total_samples)

        print(f"\nRunning benchmark on samples {start_idx} to {end_idx - 1}")
        print("=" * 80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"llada_benchmark_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        for i in tqdm(range(start_idx, end_idx), desc="Processing samples"):
            entry = dataset[i]

            # Problem statement
            problem_statement = entry.get("informal_prefix", "").strip()
            if not problem_statement:
                # Fall back to any other likely field
                problem_statement = entry.get("problem", entry.get("question", ""))

            # Generate solution
            solution, gen_time, success = self.generate_solution(problem_statement)

            # Evaluate solution quality
            quality_metrics = self.evaluate_solution_quality(
                solution,
                entry.get("formal_statement", ""),
            )

            # Print detailed output for this sample
            self.print_response_output(i, entry, solution, gen_time, success, quality_metrics)

            # Aggregate result
            result = {
                "index": i,
                "problem_id": entry.get("problem_id", f"problem_{i}"),
                "name": entry.get("name", f"Problem {i}"),
                "category": entry.get("category", "unknown"),
                "tags": entry.get("tags", []),
                "solved": entry.get("solved", False),
                "problem_statement": problem_statement,
                "formal_statement": entry.get("formal_statement", ""),
                "generated_solution": solution,
                "generation_time": gen_time,
                "generation_success": success,
                # Diffusion parameters used
                "gen_length": self.gen_length,
                "steps": self.steps,
                "block_length": self.block_length,
                "temperature": self.temperature,
                "cfg_scale": self.cfg_scale,
                "remasking": self.remasking,
                **quality_metrics,
            }

            self.results.append(result)

            # Update running stats
            self.stats["generation_times"].append(gen_time)
            self.stats["solution_lengths"].append(quality_metrics.get("solution_length", 0))
            self.stats["lean_keywords_counts"].append(quality_metrics.get("lean_keywords_used", 0))
            self.stats["categories"].append(result["category"])
            self.stats["success_rate"].append(success)

            # Save intermediate results
            if (i + 1) % save_interval == 0 or i == end_idx - 1:
                self.save_results(results_dir, f"results_batch_{i + 1}.json")
                self.print_intermediate_stats(i + 1 - start_idx)

        print(f"\nBenchmark completed! Results saved to {results_dir}/")
        return self.compile_final_report(results_dir)

    def print_intermediate_stats(self, num_processed):
        print(f"\n--- Intermediate Stats (after {num_processed} samples) ---")
        if self.stats["generation_times"]:
            avg_time = np.mean(self.stats["generation_times"])
            print(f"Average generation time: {avg_time:.2f}s")

        if self.stats["success_rate"]:
            success_rate = np.mean(self.stats["success_rate"]) * 100
            print(f"Success rate: {success_rate:.1f}%")

        if self.stats["solution_lengths"]:
            avg_length = np.mean(self.stats["solution_lengths"])
            print(f"Average solution length: {avg_length:.0f} characters")

        category_counts = Counter(self.stats["categories"])
        print(f"Top categories: {dict(list(category_counts.most_common(3)))}")

    def compile_final_report(self, results_dir):
        report = {
            "benchmark_info": {
                "model_path": self.model_path,
                "total_samples": len(self.results),
                "timestamp": datetime.now().isoformat(),
                "dataset": "MathOlympiadBench",
                "device": self.device,
                "dtype": str(self.torch_dtype),
            },
            "overall_metrics": {
                "success_rate": float(np.mean([r["generation_success"] for r in self.results]) * 100) if self.results else 0.0,
                "lean_compilation_rate": float(np.mean([r.get("lean_compilation_success", False) for r in self.results]) * 100) if self.results else 0.0,
                "average_generation_time": float(np.mean([r["generation_time"] for r in self.results])) if self.results else 0.0,
                "average_solution_length": float(np.mean([r["solution_length"] for r in self.results])) if self.results else 0.0,
                "average_lean_keywords": float(np.mean([r["lean_keywords_used"] for r in self.results])) if self.results else 0.0,
                "proof_structure_rate": float(np.mean([r["has_proof_structure"] for r in self.results]) * 100) if self.results else 0.0,
                "sorry_usage_rate": float(np.mean([r["has_sorry"] for r in self.results]) * 100) if self.results else 0.0,
                "balanced_syntax_rate": float(np.mean([r.get("has_balanced_brackets", False) for r in self.results]) * 100) if self.results else 0.0,
                "average_formal_overlap": float(np.mean([r["formal_overlap_ratio"] for r in self.results]) * 100) if self.results else 0.0,
            },
            "category_analysis": {},
        }

        # Category-wise analysis
        categories = defaultdict(list)
        for result in self.results:
            categories[result["category"]].append(result)

        for category, results in categories.items():
            if len(results) > 0:
                report["category_analysis"][category] = {
                    "count": len(results),
                    "success_rate": float(np.mean([r["generation_success"] for r in results]) * 100),
                    "avg_generation_time": float(np.mean([r["generation_time"] for r in results])),
                    "avg_solution_length": float(np.mean([r["solution_length"] for r in results])),
                    "proof_structure_rate": float(np.mean([r["has_proof_structure"] for r in results]) * 100),
                }

        # Save full report
        with open(os.path.join(results_dir, "final_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        # Save all results in one file too
        with open(os.path.join(results_dir, "all_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)

        self.print_final_summary(report)
        return report

    def print_final_summary(self, report):
        print("\n" + "=" * 80)
        print("FINAL BENCHMARK REPORT")
        print("=" * 80)

        metrics = report["overall_metrics"]
        print(f"Total samples processed: {report['benchmark_info']['total_samples']}")
        print(f"Overall success rate: {metrics['success_rate']:.1f}%")
        print(f"Lean compilation success rate: {metrics['lean_compilation_rate']:.1f}%")
        print(f"Average generation time: {metrics['average_generation_time']:.2f}s")
        print(f"Average solution length: {metrics['average_solution_length']:.0f} characters")
        print(f"Average Lean keywords used: {metrics['average_lean_keywords']:.1f}")
        print(f"Proof structure rate: {metrics['proof_structure_rate']:.1f}%")
        print(f"Sorry usage rate: {metrics['sorry_usage_rate']:.1f}%")
        print(f"Balanced syntax rate: {metrics['balanced_syntax_rate']:.1f}%")
        print(f"Average formal overlap: {metrics['average_formal_overlap']:.1f}%")

        print("\nCategory Performance:")
        for category, stats in report["category_analysis"].items():
            print(f"  {category}: {stats['success_rate']:.1f}% success ({stats['count']} samples)")

    def save_results(self, results_dir, filename):
        """Save current results to file"""
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)


# -----------------------------------------------------------------------------
# Usage example - Modified to run only 100 samples
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize benchmark with LLaDA model and diffusion parameters
    benchmark = LLaDABenchmark(
        model_path="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        device="cuda",
        torch_dtype=torch.bfloat16,
        gen_length=256,     # number of new tokens to generate
        steps=128,          # must be divisible by (gen_length / block_length)
        block_length=32,    # block size
        temperature=0.0,    # 0 for deterministic with current sampler
        cfg_scale=0.0,      # classifier-free guidance scale
        remasking="low_confidence",
        mask_id=156895,
    )

    # Run benchmark on first 100 samples only
    report = benchmark.run_benchmark(
        max_samples=100,    # Changed from None to 100
        start_idx=0,
        save_interval=25
    )

    print("Benchmark completed successfully!")
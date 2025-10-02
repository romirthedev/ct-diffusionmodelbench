import json
import os
import pandas as pd
import subprocess

class ModelAnalyzer:
    def __init__(self, base_path="./"):
        self.base_path = base_path
        self.results_paths = {
            "diffucoder": os.path.join(base_path, "Diffucoder_truebench/results_batch_100.json"),
            "dream": os.path.join(base_path, "dream_benchmark/results_batch_100.json"),
            "llada": os.path.join(base_path, "llada_benchmark/results_batch_100.json"),
        }
        self.model_specs = {
            "diffucoder": {"size": "7B", "denoising_steps": 256},
            "dream": {"size": "7B", "denoising_steps": 256},
            "llada": {"size": "7B", "denoising_steps": 128},
        }
        self.convert_script_path = os.path.join(base_path, "convert.py")

    def _load_results(self, model_name):
        with open(self.results_paths[model_name], 'r') as f:
            return json.load(f)

    def _is_valid_lean(self, lean_code):
        # This function will call the convert.py script to validate Lean syntax
        try:
            # Skip validation for obviously malformed code to prevent hanging
            if not lean_code.strip() or len(lean_code) > 10000:
                return False
            
            # Extract Lean code from markdown code blocks if present
            clean_code = lean_code.strip()
            if clean_code.startswith('```lean'):
                lines = clean_code.split('\n')
                # Remove first line (```lean) and last line (```) if present
                if len(lines) > 1:
                    clean_code = '\n'.join(lines[1:])
                    if clean_code.endswith('```'):
                        clean_code = clean_code[:-3].strip()
            elif clean_code.startswith('```'):
                lines = clean_code.split('\n')
                if len(lines) > 1:
                    clean_code = '\n'.join(lines[1:])
                    if clean_code.endswith('```'):
                        clean_code = clean_code[:-3].strip()
                
            process = subprocess.run(
                ["python", self.convert_script_path, "--check_lean_syntax", clean_code],
                capture_output=True, text=True, timeout=10
            )
            return process.returncode == 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Silently return False for validation errors to avoid spam
            return False
        except Exception as e:
            return False

    def analyze_model(self, model_name):
        print(f"Analyzing {model_name}...")
        results = self._load_results(model_name)

        total_solutions = len(results)
        successful_generations = sum(1 for r in results if r.get("generated_solution"))
        compilation_successes = sum(1 for r in results if r.get("lean_compilation_success"))
        valid_lean_syntax = sum(1 for r in results if self._is_valid_lean(r.get("generated_solution", "")))
        has_proof_structure = sum(1 for r in results if r.get("has_proof_structure"))
        avg_generation_time = sum(r.get("generation_time", 0) for r in results) / total_solutions if total_solutions > 0 else 0
        lean_keywords_used = sum(r.get("lean_keywords_used", 0) for r in results)

        # Meta-analysis of Lean syntax knowledge
        theorems_used = sum(1 for r in results if r.get("has_theorem_declaration"))
        return {
            "model": model_name,
            "size": self.model_specs[model_name]["size"],
            "denoising_steps": self.model_specs[model_name]["denoising_steps"],
            "total_solutions": total_solutions,
            "successful_generations": successful_generations,
            "compilation_success_rate": (compilation_successes / total_solutions) * 100 if total_solutions > 0 else 0,
            "valid_lean_syntax_rate": (valid_lean_syntax / total_solutions) * 100 if total_solutions > 0 else 0,
            "proof_structure_usage": (has_proof_structure / total_solutions) * 100 if total_solutions > 0 else 0,
            "avg_generation_time": avg_generation_time,
            "avg_lean_keywords_used": lean_keywords_used / total_solutions if total_solutions > 0 else 0,
            "theorems_used": theorems_used,
        }

    def run_analysis(self):
        all_results = []
        for model_name in self.results_paths.keys():
            all_results.append(self.analyze_model(model_name))

        df = pd.DataFrame(all_results)
        print("\n--- Performance Summary ---")
        print(df.to_markdown(index=False))

        # Save to various formats
        output_dir = os.path.join(self.base_path, "processing")
        os.makedirs(output_dir, exist_ok=True)

        df.to_csv(os.path.join(output_dir, "model_performance.csv"), index=False)
        df.to_excel(os.path.join(output_dir, "model_performance.xlsx"), index=False)
        with open(os.path.join(output_dir, "model_performance.md"), "w") as f:
            f.write(df.to_markdown(index=False))
        with open(os.path.join(output_dir, "model_performance.json"), "w") as f:
            json.dump(all_results, f, indent=4)

        print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    analyzer = ModelAnalyzer(base_path="/Users/romirpatel/ct-diffusionmodelbench/")
    analyzer.run_analysis()
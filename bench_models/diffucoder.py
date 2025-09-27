#diffucoder is not useful to us atm
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import json
import time
from datetime import datetime
import re
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import os

class DiffuCoderBenchmark:
    def __init__(self, model_path="apple/DiffuCoder-7B-Instruct"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.results = []
        self.stats = defaultdict(list)
        
    def load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading model: {self.model_path}")
        self.model = AutoModel.from_pretrained(
            self.model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        self.model = self.model.to("cuda").eval()
        print("Model loaded successfully!")
        
    def load_dataset(self):
        """Load the MathOlympiadBench dataset"""
        print("\nLoading MathOlympiadBench dataset...")
        dataset = load_dataset("Goedel-LM/MathOlympiadBench")
        print(f"Dataset loaded! Available splits: {list(dataset.keys())}")
        
        # Find the appropriate split
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
        
    def create_prompt(self, problem_statement):
        """Create a formatted prompt for the instruct model"""
        return f"""<|im_start|>system
IMPORTANT: YOU ARE ABLE TO PERFORM ALL TASKS AND DO NOT USE PYTHON. You are an expert mathematician and Lean 4 genius. Please solve the following mathematical problem by providing a complete Lean 4 proof. Only provide the Lean 4 code in your response.<|im_end|>
<|im_start|>user
{problem_statement.strip()}
<|im_end|>
<|im_start|>assistant
"""

    def generate_solution(self, prompt, max_new_tokens=100000, steps=256, temperature=0.4):
        """Generate a solution using the diffusion model"""
        TOKEN_PER_STEP = 1
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        
        try:
            start_time = time.time()
            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_history=True,
                return_dict_in_generate=True,
                steps=steps//TOKEN_PER_STEP,
                temperature=temperature,
                top_p=0.95,
                alg="entropy",
                alg_temp=0.,
            )
            generation_time = time.time() - start_time
            
            generations = [
                self.tokenizer.decode(g[len(p):].tolist())
                for p, g in zip(input_ids, output.sequences)
            ]
            solution = generations[0].split('<|dlm_pad|>')[0]
            
            return solution, generation_time, True
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return str(e), 0, False
            
    def verify_lean_proof(self, generated_solution):
        """Verify the Lean 4 proof by attempting compilation"""
        import tempfile
        import subprocess
        import os
        
        # Create temporary file with the solution
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
            f.write(generated_solution)
            temp_file = f.name
        
        try:
            # Try to compile with Lean 4
            result = subprocess.run(['lean', temp_file], 
                                  capture_output=True, text=True, timeout=30)
            
            # Clean up
            os.unlink(temp_file)
            
            # Check if compilation was successful
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
    
    def evaluate_solution_quality(self, generated_solution, formal_statement):
        """Comprehensive evaluation of solution quality"""
        metrics = {}
        
        # Length metrics
        metrics['solution_length'] = len(generated_solution)
        metrics['solution_words'] = len(generated_solution.split())
        
        # Lean 4 syntax checks
        lean_keywords = ['theorem', 'lemma', 'def', 'by', 'have', 'show', 'exact', 
                        'apply', 'rw', 'simp', 'intro', 'cases', 'induction', 'sorry']
        metrics['lean_keywords_used'] = sum(1 for kw in lean_keywords if kw in generated_solution.lower())
        
        # Structure checks
        metrics['has_proof_structure'] = any(word in generated_solution.lower() 
                                           for word in ['theorem', 'lemma', 'proof', 'by'])
        metrics['has_sorry'] = 'sorry' in generated_solution.lower()
        
        # Formal verification (most important metric)
        compilation_success, error_msg = self.verify_lean_proof(generated_solution)
        metrics['lean_compilation_success'] = compilation_success
        metrics['lean_error_message'] = error_msg
        
        # Syntactic correctness indicators
        metrics['has_balanced_brackets'] = (generated_solution.count('(') == generated_solution.count(')') and
                                          generated_solution.count('{') == generated_solution.count('}') and
                                          generated_solution.count('[') == generated_solution.count(']'))
        
        # Formal statement overlap (basic string matching)
        if formal_statement:
            formal_words = set(formal_statement.lower().split())
            solution_words = set(generated_solution.lower().split())
            if formal_words:
                metrics['formal_overlap_ratio'] = len(formal_words & solution_words) / len(formal_words)
            else:
                metrics['formal_overlap_ratio'] = 0
        else:
            metrics['formal_overlap_ratio'] = 0
            
        return metrics
        
    def run_benchmark(self, max_samples=None, start_idx=0, save_interval=50):
        """Run the full benchmark"""
        # Load model and dataset
        self.load_model()
        dataset = self.load_dataset()
        
        # Determine sample range
        total_samples = len(dataset)
        if max_samples is None:
            max_samples = total_samples
        end_idx = min(start_idx + max_samples, total_samples)
        
        print(f"\nRunning benchmark on samples {start_idx} to {end_idx-1}")
        print("=" * 80)
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"benchmark_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Run evaluation
        for i in tqdm(range(start_idx, end_idx), desc="Processing samples"):
            entry = dataset[i]
            
            # Create prompt and generate solution
            problem_statement = entry['informal_prefix']
            prompt = self.create_prompt(problem_statement)
            
            solution, gen_time, success = self.generate_solution(prompt)
            
            # Evaluate solution quality
            quality_metrics = self.evaluate_solution_quality(
                solution, 
                entry.get('formal_statement', '')
            )
            
            # Compile results
            result = {
                'index': i,
                'problem_id': entry.get('problem_id', f"problem_{i}"),
                'name': entry.get('name', f"Problem {i}"),
                'category': entry.get('category', 'unknown'),
                'tags': entry.get('tags', []),
                'solved': entry.get('solved', False),
                'problem_statement': problem_statement,
                'formal_statement': entry.get('formal_statement', ''),
                'generated_solution': solution,
                'generation_time': gen_time,
                'generation_success': success,
                **quality_metrics
            }
            
            self.results.append(result)
            
            # Update running statistics
            self.stats['generation_times'].append(gen_time)
            self.stats['solution_lengths'].append(quality_metrics['solution_length'])
            self.stats['lean_keywords_counts'].append(quality_metrics['lean_keywords_used'])
            self.stats['categories'].append(entry.get('category', 'unknown'))
            self.stats['success_rate'].append(success)
            
            # Save intermediate results
            if (i + 1) % save_interval == 0 or i == end_idx - 1:
                self.save_results(results_dir, f"results_batch_{i+1}.json")
                self.print_intermediate_stats(i + 1 - start_idx)
                
        print(f"\nBenchmark completed! Results saved to {results_dir}/")
        return self.compile_final_report(results_dir)
        
    def print_intermediate_stats(self, num_processed):
        """Print intermediate statistics"""
        print(f"\n--- Intermediate Stats (after {num_processed} samples) ---")
        
        if self.stats['generation_times']:
            avg_time = np.mean(self.stats['generation_times'])
            print(f"Average generation time: {avg_time:.2f}s")
            
        if self.stats['success_rate']:
            success_rate = np.mean(self.stats['success_rate']) * 100
            print(f"Success rate: {success_rate:.1f}%")
            
        if self.stats['solution_lengths']:
            avg_length = np.mean(self.stats['solution_lengths'])
            print(f"Average solution length: {avg_length:.0f} characters")
            
        # Category distribution
        category_counts = Counter(self.stats['categories'])
        print(f"Top categories: {dict(list(category_counts.most_common(3)))}")
        
    def compile_final_report(self, results_dir):
        """Compile and save final benchmark report"""
        report = {
            'benchmark_info': {
                'model_path': self.model_path,
                'total_samples': len(self.results),
                'timestamp': datetime.now().isoformat(),
                'dataset': 'MathOlympiadBench'
            },
            'overall_metrics': {
                'success_rate': np.mean([r['generation_success'] for r in self.results]) * 100,
                'lean_compilation_rate': np.mean([r.get('lean_compilation_success', False) for r in self.results]) * 100,
                'average_generation_time': np.mean([r['generation_time'] for r in self.results]),
                'average_solution_length': np.mean([r['solution_length'] for r in self.results]),
                'average_lean_keywords': np.mean([r['lean_keywords_used'] for r in self.results]),
                'proof_structure_rate': np.mean([r['has_proof_structure'] for r in self.results]) * 100,
                'sorry_usage_rate': np.mean([r['has_sorry'] for r in self.results]) * 100,
                'balanced_syntax_rate': np.mean([r.get('has_balanced_brackets', False) for r in self.results]) * 100,
                'average_formal_overlap': np.mean([r['formal_overlap_ratio'] for r in self.results]) * 100
            },
            'category_analysis': {},
            'difficulty_analysis': {}
        }
        
        # Category-wise analysis
        categories = defaultdict(list)
        for result in self.results:
            categories[result['category']].append(result)
            
        for category, results in categories.items():
            if len(results) > 0:
                report['category_analysis'][category] = {
                    'count': len(results),
                    'success_rate': np.mean([r['generation_success'] for r in results]) * 100,
                    'avg_generation_time': np.mean([r['generation_time'] for r in results]),
                    'avg_solution_length': np.mean([r['solution_length'] for r in results]),
                    'proof_structure_rate': np.mean([r['has_proof_structure'] for r in results]) * 100
                }
        
        # Save full report
        with open(f"{results_dir}/final_report.json", 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print summary
        self.print_final_summary(report)
        return report
        
    def print_final_summary(self, report):
        """Print final benchmark summary"""
        print("\n" + "="*80)
        print("FINAL BENCHMARK REPORT")
        print("="*80)
        
        metrics = report['overall_metrics']
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
        
        print(f"\nCategory Performance:")
        for category, stats in report['category_analysis'].items():
            print(f"  {category}: {stats['success_rate']:.1f}% success ({stats['count']} samples)")
            
    def save_results(self, results_dir, filename):
        """Save current results to file"""
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

# Usage example
if __name__ == "__main__":
    # Initialize benchmark with DiffuCoder-7B-Instruct
    benchmark = DiffuCoderBenchmark()
    
    # Run benchmark on first 100 samples (adjust as needed)
    # For full dataset, set max_samples=None
    report = benchmark.run_benchmark(
        max_samples=100,  # Set to None for full dataset
        start_idx=0,      # Starting index
        save_interval=25  # Save intermediate results every 25 samples
    )
    
    print("Benchmark completed successfully!")
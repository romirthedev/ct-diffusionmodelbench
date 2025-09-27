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
import gc
import tempfile
import subprocess

class DreamCoderBenchmark:
    def __init__(self, model_path="Dream-org/Dream-Coder-v0-Instruct-7B"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.results = []
        self.stats = defaultdict(list)
        
    def load_model(self):
        """Load the Dream-Coder model and tokenizer"""
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
        print("Dream-Coder model loaded successfully!")
        
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
        """Create a formatted prompt for Dream-Coder"""
        return f"""<|im_start|>system
You are an expert mathematician and Lean 4 programmer. Please solve the following mathematical problem by providing a complete Lean 4 proof. Only provide the Lean 4 code in your response. IMPORTANT: DO NOT provide ANYTHING ELSE. Provide full Lean4 solution only.<|im_end|>
<|im_start|>user
{problem_statement.strip()}
<|im_end|>
<|im_start|>assistant
"""

    def generate_solution(self, prompt, max_new_tokens=4096, steps=256, temperature=0.4):
        """Generate a solution using Dream-Coder's diffusion generation"""
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
                steps=steps // TOKEN_PER_STEP,
                temperature=temperature,
                top_p=0.95,
                alg="entropy",
                alg_temp=0.0,
            )
            generation_time = time.time() - start_time
            
            # Decode output (Dream-Coder specific)
            generations = [
                self.tokenizer.decode(g[len(p):].tolist())
                for p, g in zip(input_ids, output.sequences)
            ]
            solution = generations[0].split(self.tokenizer.eos_token)[0]
            
            return solution, generation_time, True
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return str(e), 0, False
        finally:
            # Clean up GPU memory
            if 'input_ids' in locals():
                del input_ids, attention_mask
            torch.cuda.empty_cache()

    def verify_lean_proof(self, generated_solution):
        """Verify the Lean 4 proof by attempting compilation"""
        # Ensure Lean is in PATH (important for TACC/cluster environments)
        lean_path = os.path.expanduser("~/.elan/bin/lean")
        if not os.path.exists(lean_path):
            lean_path = "lean"  # Fallback to system PATH
        
        # Create temporary file with the solution
        # Add basic Lean 4 imports that are commonly needed
        lean_code = f"""-- Auto-generated proof verification
{generated_solution}
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
            f.write(lean_code)
            temp_file = f.name
        
        try:
            # Set up environment for subprocess
            env = os.environ.copy()
            env['PATH'] = f"{os.path.expanduser('~/.elan/bin')}:{env.get('PATH', '')}"
            
            # Try to compile with Lean 4
            result = subprocess.run([lean_path, temp_file], 
                                  capture_output=True, text=True, timeout=60, env=env)
            
            # Clean up
            os.unlink(temp_file)
            
            # Check if compilation was successful
            compilation_success = result.returncode == 0
            error_messages = result.stderr if result.stderr else ""
            
            # Log successful compilations for debugging
            if compilation_success:
                print(f"✓ Lean compilation successful")
            else:
                print(f"✗ Lean compilation failed: {error_messages[:200]}...")
            
            return compilation_success, error_messages
            
        except subprocess.TimeoutExpired:
            os.unlink(temp_file)
            return False, "Compilation timeout (60s)"
        except FileNotFoundError:
            os.unlink(temp_file)
            return False, f"Lean 4 not found at {lean_path} - check Elan installation"
        except Exception as e:
            os.unlink(temp_file)
            return False, f"Verification error: {str(e)}"
            
    def evaluate_solution_quality(self, generated_solution, formal_statement, problem_statement):
        """Comprehensive evaluation of solution quality"""
        metrics = {}
        
        # Basic length and structure metrics
        metrics['solution_length'] = len(generated_solution)
        metrics['solution_words'] = len(generated_solution.split())
        metrics['solution_lines'] = len(generated_solution.split('\n'))
        
        # Lean 4 specific syntax and keyword analysis
        lean_keywords = [
            'theorem', 'lemma', 'def', 'by', 'have', 'show', 'exact', 'apply', 
            'rw', 'simp', 'intro', 'cases', 'induction', 'sorry', 'qed',
            'calc', 'obtain', 'use', 'refine', 'constructor', 'left', 'right',
            'exists', 'forall', 'fun', 'let', 'where', 'match', 'with'
        ]
        
        lean_tactics = [
            'simp', 'rw', 'apply', 'exact', 'intro', 'cases', 'induction',
            'constructor', 'left', 'right', 'split', 'use', 'existsi', 
            'refine', 'calc', 'ring', 'field_simp', 'norm_num', 'linarith'
        ]
        
        solution_lower = generated_solution.lower()
        metrics['lean_keywords_used'] = sum(1 for kw in lean_keywords if kw in solution_lower)
        metrics['lean_tactics_used'] = sum(1 for tac in lean_tactics if tac in solution_lower)
        
        # Structure and completeness checks
        metrics['has_theorem_declaration'] = any(word in solution_lower for word in ['theorem', 'lemma'])
        metrics['has_proof_structure'] = any(word in solution_lower for word in ['by', 'proof', ':='])
        metrics['has_sorry'] = 'sorry' in solution_lower
        metrics['has_qed'] = any(word in solution_lower for word in ['qed', 'done'])
        
        # Formal verification (most important metric)
        compilation_success, error_msg = self.verify_lean_proof(generated_solution)
        metrics['lean_compilation_success'] = compilation_success
        metrics['lean_error_message'] = error_msg
        
        # Mathematical content analysis
        math_symbols = ['∀', '∃', '→', '↔', '∧', '∨', '¬', '≤', '≥', '≠', '∈', '⊆', '∪', '∩']
        metrics['math_symbols_count'] = sum(solution_lower.count(symbol.lower()) for symbol in math_symbols)
        
        # Code structure analysis
        metrics['has_imports'] = 'import' in solution_lower
        metrics['has_variables'] = 'variable' in solution_lower or 'var' in solution_lower
        metrics['has_balanced_brackets'] = (generated_solution.count('(') == generated_solution.count(')') and
                                          generated_solution.count('{') == generated_solution.count('}') and
                                          generated_solution.count('[') == generated_solution.count(']'))
        
        # Formal statement overlap analysis
        if formal_statement and len(formal_statement.strip()) > 0:
            formal_words = set(re.findall(r'\w+', formal_statement.lower()))
            solution_words = set(re.findall(r'\w+', solution_lower))
            if formal_words:
                metrics['formal_overlap_ratio'] = len(formal_words & solution_words) / len(formal_words)
                metrics['formal_unique_words'] = len(formal_words - solution_words)
            else:
                metrics['formal_overlap_ratio'] = 0
                metrics['formal_unique_words'] = 0
        else:
            metrics['formal_overlap_ratio'] = 0
            metrics['formal_unique_words'] = 0
            
        # Problem statement relevance
        if problem_statement and len(problem_statement.strip()) > 0:
            problem_words = set(re.findall(r'\w+', problem_statement.lower()))
            solution_words = set(re.findall(r'\w+', solution_lower))
            if problem_words:
                metrics['problem_overlap_ratio'] = len(problem_words & solution_words) / len(problem_words)
            else:
                metrics['problem_overlap_ratio'] = 0
        else:
            metrics['problem_overlap_ratio'] = 0
            
        # Quality heuristics
        metrics['appears_complete'] = (
            metrics['has_theorem_declaration'] and 
            metrics['has_proof_structure'] and 
            not metrics['has_sorry'] and
            metrics['solution_length'] > 50
        )
        
        metrics['complexity_score'] = (
            metrics['lean_tactics_used'] * 2 +
            metrics['math_symbols_count'] +
            metrics['solution_lines'] * 0.5
        )
        
        return metrics
        
    def convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
        
    def run_benchmark(self, max_samples=None, start_idx=0, save_interval=50, 
                     max_new_tokens=256, temperature=0.4):
        """Run the comprehensive benchmark"""
        # Load model and dataset
        self.load_model()
        dataset = self.load_dataset()
        
        # Determine sample range
        total_samples = len(dataset)
        if max_samples is None:
            max_samples = total_samples
        end_idx = min(start_idx + max_samples, total_samples)
        
        print(f"\nRunning Dream-Coder benchmark on samples {start_idx} to {end_idx-1}")
        print(f"Generation parameters: max_tokens={max_new_tokens}, temperature={temperature}")
        print("=" * 80)
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"dreamcoder_benchmark_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize tracking variables
        failed_generations = []
        category_stats = defaultdict(list)
        
        # Run evaluation
        for i in tqdm(range(start_idx, end_idx), desc="Processing samples"):
            entry = dataset[i]
            
            # Extract problem information
            problem_statement = entry['informal_prefix']
            formal_statement = entry.get('formal_statement', '')
            
            # Create prompt and generate solution
            prompt = self.create_prompt(problem_statement)
            solution, gen_time, success = self.generate_solution(
                prompt, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature
            )
            
            # Evaluate solution quality
            quality_metrics = self.evaluate_solution_quality(
                solution, formal_statement, problem_statement
            )
            
            # Compile comprehensive results
            result = {
                'index': i,
                'problem_id': entry.get('problem_id', f"problem_{i}"),
                'name': entry.get('name', f"Problem {i}"),
                'category': entry.get('category', 'unknown'),
                'tags': entry.get('tags', []),
                'solved': entry.get('solved', False),
                'difficulty': entry.get('difficulty', 'unknown'),
                'problem_statement': problem_statement,
                'formal_statement': formal_statement,
                'generated_solution': solution,
                'generation_time': gen_time,
                'generation_success': success,
                'prompt_length': len(prompt),
                **quality_metrics
            }
            
            self.results.append(result)
            
            # Track failures for analysis
            if not success:
                failed_generations.append({
                    'index': i,
                    'error': solution,
                    'category': entry.get('category', 'unknown')
                })
            
            # Update running statistics
            self.update_running_stats(result, entry)
            category_stats[entry.get('category', 'unknown')].append(result)
            
            # Save intermediate results and print stats
            if (i + 1) % save_interval == 0 or i == end_idx - 1:
                self.save_intermediate_results(results_dir, i + 1, failed_generations)
                self.print_intermediate_stats(i + 1 - start_idx, category_stats)
                
        print(f"\nBenchmark completed! Results saved to {results_dir}/")
        return self.compile_final_report(results_dir, failed_generations, category_stats)
        
    def update_running_stats(self, result, entry):
        """Update running statistics"""
        self.stats['generation_times'].append(result['generation_time'])
        self.stats['solution_lengths'].append(result['solution_length'])
        self.stats['lean_keywords_counts'].append(result['lean_keywords_used'])
        self.stats['lean_tactics_counts'].append(result['lean_tactics_used'])
        self.stats['categories'].append(result['category'])
        self.stats['success_rate'].append(result['generation_success'])
        self.stats['completeness_rate'].append(result['appears_complete'])
        self.stats['complexity_scores'].append(result['complexity_score'])
        self.stats['formal_overlaps'].append(result['formal_overlap_ratio'])
        self.stats['compilation_success'].append(result.get('lean_compilation_success', False))
        
    def print_intermediate_stats(self, num_processed, category_stats):
        """Print comprehensive intermediate statistics"""
        print(f"\n--- Intermediate Stats (after {num_processed} samples) ---")
        
        if self.stats['generation_times']:
            avg_time = np.mean(self.stats['generation_times'])
            print(f"Average generation time: {avg_time:.2f}s")
            
        if self.stats['success_rate']:
            success_rate = np.mean(self.stats['success_rate']) * 100
            print(f"Generation success rate: {success_rate:.1f}%")
            
        if self.stats['compilation_success']:
            compilation_rate = np.mean(self.stats['compilation_success']) * 100
            print(f"Lean compilation success rate: {compilation_rate:.1f}%")
            
        if self.stats['completeness_rate']:
            completeness_rate = np.mean(self.stats['completeness_rate']) * 100
            print(f"Apparent completeness rate: {completeness_rate:.1f}%")
            
        if self.stats['solution_lengths']:
            avg_length = np.mean(self.stats['solution_lengths'])
            median_length = np.median(self.stats['solution_lengths'])
            print(f"Solution length - Avg: {avg_length:.0f}, Median: {median_length:.0f} chars")
            
        if self.stats['lean_keywords_counts']:
            avg_keywords = np.mean(self.stats['lean_keywords_counts'])
            avg_tactics = np.mean(self.stats['lean_tactics_counts'])
            print(f"Avg Lean keywords: {avg_keywords:.1f}, Avg tactics: {avg_tactics:.1f}")
            
        if self.stats['complexity_scores']:
            avg_complexity = np.mean(self.stats['complexity_scores'])
            print(f"Average complexity score: {avg_complexity:.1f}")
            
        # Top performing categories
        category_success = {}
        for cat, results in category_stats.items():
            if len(results) >= 3:  # Only show categories with enough samples
                success_rate = np.mean([r['generation_success'] for r in results]) * 100
                category_success[cat] = success_rate
        
        if category_success:
            top_categories = sorted(category_success.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"Top categories by success: {dict(top_categories)}")
        
    def compile_final_report(self, results_dir, failed_generations, category_stats):
        """Compile comprehensive final benchmark report"""
        report = {
            'benchmark_info': {
                'model_path': self.model_path,
                'model_name': 'Dream-Coder-v0-Instruct-7B',
                'total_samples': len(self.results),
                'timestamp': datetime.now().isoformat(),
                'dataset': 'MathOlympiadBench'
            },
            'overall_metrics': self.calculate_overall_metrics(),
            'category_analysis': self.analyze_by_category(category_stats),
            'quality_analysis': self.analyze_solution_quality(),
            'failure_analysis': self.analyze_failures(failed_generations),
            'detailed_statistics': self.calculate_detailed_stats()
        }
        
        # Convert numpy types to native Python types for JSON serialization
        report = self.convert_numpy_types(report)
        
        # Save full report
        with open(f"{results_dir}/final_report.json", 'w') as f:
            json.dump(report, f, indent=2)
            
        # Save summary report
        summary = self.create_summary_report(report)
        with open(f"{results_dir}/summary_report.txt", 'w') as f:
            f.write(summary)
            
        # Print final summary
        print(summary)
        return report
        
    def calculate_overall_metrics(self):
        """Calculate comprehensive overall metrics"""
        if not self.results:
            return {}
            
        return {
            'generation_success_rate': np.mean([r['generation_success'] for r in self.results]) * 100,
            'lean_compilation_rate': np.mean([r.get('lean_compilation_success', False) for r in self.results]) * 100,
            'apparent_completeness_rate': np.mean([r['appears_complete'] for r in self.results]) * 100,
            'average_generation_time': np.mean([r['generation_time'] for r in self.results]),
            'median_generation_time': np.median([r['generation_time'] for r in self.results]),
            'average_solution_length': np.mean([r['solution_length'] for r in self.results]),
            'median_solution_length': np.median([r['solution_length'] for r in self.results]),
            'average_lean_keywords': np.mean([r['lean_keywords_used'] for r in self.results]),
            'average_lean_tactics': np.mean([r['lean_tactics_used'] for r in self.results]),
            'theorem_declaration_rate': np.mean([r['has_theorem_declaration'] for r in self.results]) * 100,
            'proof_structure_rate': np.mean([r['has_proof_structure'] for r in self.results]) * 100,
            'sorry_usage_rate': np.mean([r['has_sorry'] for r in self.results]) * 100,
            'balanced_syntax_rate': np.mean([r.get('has_balanced_brackets', False) for r in self.results]) * 100,
            'average_formal_overlap': np.mean([r['formal_overlap_ratio'] for r in self.results]) * 100,
            'average_problem_overlap': np.mean([r['problem_overlap_ratio'] for r in self.results]) * 100,
            'average_complexity_score': np.mean([r['complexity_score'] for r in self.results])
        }
        
    def analyze_by_category(self, category_stats):
        """Analyze performance by problem category"""
        analysis = {}
        for category, results in category_stats.items():
            if len(results) > 0:
                analysis[category] = {
                    'count': len(results),
                    'success_rate': np.mean([r['generation_success'] for r in results]) * 100,
                    'compilation_rate': np.mean([r.get('lean_compilation_success', False) for r in results]) * 100,
                    'completeness_rate': np.mean([r['appears_complete'] for r in results]) * 100,
                    'avg_generation_time': np.mean([r['generation_time'] for r in results]),
                    'avg_solution_length': np.mean([r['solution_length'] for r in results]),
                    'avg_complexity_score': np.mean([r['complexity_score'] for r in results]),
                    'theorem_rate': np.mean([r['has_theorem_declaration'] for r in results]) * 100,
                    'sorry_rate': np.mean([r['has_sorry'] for r in results]) * 100
                }
        return analysis
        
    def analyze_solution_quality(self):
        """Analyze solution quality patterns"""
        if not self.results:
            return {}
            
        successful_results = [r for r in self.results if r['generation_success']]
        complete_results = [r for r in self.results if r['appears_complete']]
        compiled_results = [r for r in self.results if r.get('lean_compilation_success', False)]
        
        return {
            'solutions_with_theorems': sum(1 for r in self.results if r['has_theorem_declaration']),
            'solutions_with_proofs': sum(1 for r in self.results if r['has_proof_structure']),
            'solutions_with_sorry': sum(1 for r in self.results if r['has_sorry']),
            'solutions_compiled': len(compiled_results),
            'empty_solutions': sum(1 for r in self.results if r['solution_length'] < 10),
            'high_quality_solutions': len(complete_results),
            'avg_keywords_in_complete': np.mean([r['lean_keywords_used'] for r in complete_results]) if complete_results else 0,
            'avg_tactics_in_complete': np.mean([r['lean_tactics_used'] for r in complete_results]) if complete_results else 0,
            'avg_keywords_in_compiled': np.mean([r['lean_keywords_used'] for r in compiled_results]) if compiled_results else 0,
            'complexity_distribution': {
                'low': sum(1 for r in self.results if r['complexity_score'] < 5),
                'medium': sum(1 for r in self.results if 5 <= r['complexity_score'] < 15),
                'high': sum(1 for r in self.results if r['complexity_score'] >= 15)
            }
        }
        
    def analyze_failures(self, failed_generations):
        """Analyze generation failures"""
        if not failed_generations:
            return {'total_failures': 0}
            
        failure_categories = Counter(f['category'] for f in failed_generations)
        
        return {
            'total_failures': len(failed_generations),
            'failure_rate': len(failed_generations) / len(self.results) * 100 if self.results else 0,
            'failures_by_category': dict(failure_categories),
            'common_error_patterns': self.extract_error_patterns(failed_generations)
        }
        
    def extract_error_patterns(self, failed_generations):
        """Extract common error patterns from failures"""
        error_patterns = Counter()
        for failure in failed_generations:
            error = failure['error'].lower()
            if 'cuda' in error or 'memory' in error:
                error_patterns['memory_issues'] += 1
            elif 'timeout' in error:
                error_patterns['timeout'] += 1
            elif 'shape' in error or 'dimension' in error:
                error_patterns['tensor_shape'] += 1
            else:
                error_patterns['other'] += 1
        return dict(error_patterns)
        
    def calculate_detailed_stats(self):
        """Calculate detailed statistical information"""
        if not self.results:
            return {}
            
        gen_times = [r['generation_time'] for r in self.results]
        solution_lengths = [r['solution_length'] for r in self.results]
        complexity_scores = [r['complexity_score'] for r in self.results]
        
        return {
            'generation_time_stats': {
                'min': float(np.min(gen_times)),
                'max': float(np.max(gen_times)),
                'std': float(np.std(gen_times)),
                'percentile_95': float(np.percentile(gen_times, 95))
            },
            'solution_length_stats': {
                'min': int(np.min(solution_lengths)),
                'max': int(np.max(solution_lengths)),
                'std': float(np.std(solution_lengths)),
                'percentile_95': float(np.percentile(solution_lengths, 95))
            },
            'complexity_score_stats': {
                'min': float(np.min(complexity_scores)),
                'max': float(np.max(complexity_scores)),
                'std': float(np.std(complexity_scores)),
                'percentile_95': float(np.percentile(complexity_scores, 95))
            }
        }
        
    def create_summary_report(self, report):
        """Create a human-readable summary report"""
        metrics = report['overall_metrics']
        
        summary = f"""
Dream-Coder MathOlympiadBench Benchmark Summary
{'=' * 60}

Model: {report['benchmark_info']['model_name']}
Dataset: {report['benchmark_info']['dataset']}
Total Samples: {report['benchmark_info']['total_samples']}
Timestamp: {report['benchmark_info']['timestamp']}

OVERALL PERFORMANCE
{'=' * 30}
Generation Success Rate: {metrics['generation_success_rate']:.1f}%
Lean Compilation Success Rate: {metrics['lean_compilation_rate']:.1f}%
Apparent Completeness Rate: {metrics['apparent_completeness_rate']:.1f}%
Average Generation Time: {metrics['average_generation_time']:.2f}s

SOLUTION QUALITY
{'=' * 30}
Average Solution Length: {metrics['average_solution_length']:.0f} characters
Average Lean Keywords Used: {metrics['average_lean_keywords']:.1f}
Average Lean Tactics Used: {metrics['average_lean_tactics']:.1f}
Theorem Declaration Rate: {metrics['theorem_declaration_rate']:.1f}%
Proof Structure Rate: {metrics['proof_structure_rate']:.1f}%
Sorry Usage Rate: {metrics['sorry_usage_rate']:.1f}%
Balanced Syntax Rate: {metrics['balanced_syntax_rate']:.1f}%

CONTENT ANALYSIS
{'=' * 30}
Average Formal Overlap: {metrics['average_formal_overlap']:.1f}%
Average Problem Overlap: {metrics['average_problem_overlap']:.1f}%
Average Complexity Score: {metrics['average_complexity_score']:.1f}

TOP CATEGORIES BY SUCCESS RATE
{'=' * 30}"""
        
        # Add top categories
        if 'category_analysis' in report:
            sorted_categories = sorted(
                report['category_analysis'].items(),
                key=lambda x: x[1]['success_rate'],
                reverse=True
            )[:5]
            
            for cat, stats in sorted_categories:
                summary += f"\n{cat}: {stats['success_rate']:.1f}% success, {stats['compilation_rate']:.1f}% compiled ({stats['count']} samples)"
        
        if 'failure_analysis' in report and report['failure_analysis']['total_failures'] > 0:
            summary += f"""

FAILURE ANALYSIS
{'=' * 30}
Total Failures: {report['failure_analysis']['total_failures']}
Failure Rate: {report['failure_analysis']['failure_rate']:.1f}%
"""
        
        summary += "\n" + "=" * 60
        return summary
        
    def save_intermediate_results(self, results_dir, current_idx, failed_generations):
        """Save intermediate results and failure logs"""
        # Save current results
        with open(f"{results_dir}/results_batch_{current_idx}.json", 'w') as f:
            json.dump(self.convert_numpy_types(self.results), f, indent=2)
            
        # Save failure log
        if failed_generations:
            with open(f"{results_dir}/failures_{current_idx}.json", 'w') as f:
                json.dump(failed_generations, f, indent=2)

if __name__ == "__main__":
    # Initialize benchmark
    benchmark = DreamCoderBenchmark()
    
    # Configuration options
    config = {
        'max_samples': 100,        # Set to None for full dataset
        'start_idx': 0,            # Starting index
        'save_interval': 25,       # Save intermediate results every N samples
        'max_new_tokens': 4096,     # Maximum tokens to generate
        'temperature': 0.4         # Generation temperature
    }
    
    print("Starting Dream-Coder MathOlympiadBench Benchmark...")
    print(f"Configuration: {config}")
    
    # Run benchmark
    try:
        report = benchmark.run_benchmark(**config)
        print("\nBenchmark completed successfully!")
        print(f"Results saved with {len(benchmark.results)} samples processed.")
        
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        print("Partial results may have been saved.")
    
    finally:
        # Clean up GPU memory
        if benchmark.model is not None:
            del benchmark.model
            if hasattr(benchmark, 'tokenizer'):
                del benchmark.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
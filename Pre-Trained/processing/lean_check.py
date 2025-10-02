import json
import subprocess
import sys
import tempfile
import os

def is_valid_lean(snippet: str) -> bool:
    if not snippet.strip():
        return False

    # Write snippet to a temporary lean file
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, "snippet.lean")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(snippet)

        # Run lean to check syntax
        try:
            result = subprocess.run(
                ["lean", fn],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

def main(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level array")

    valid_count = 0
    for item in data:
        snippet = str(item.get("generated_solution", ""))
        if is_valid_lean(snippet):
            valid_count += 1

    print(f"Total items: {len(data)}")
    print(f"Items with syntactically valid Lean4: {valid_count}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert.py path/to/json OR python3 convert.py --check_lean_syntax 'code'")
        sys.exit(1)
    
    if sys.argv[1] == "--check_lean_syntax":
        if len(sys.argv) < 3:
            print("Error: No code snippet provided")
            sys.exit(1)
        snippet = sys.argv[2]
        if is_valid_lean(snippet):
            print("Valid Lean syntax")
            sys.exit(0)
        else:
            print("Invalid Lean syntax")
            sys.exit(1)
    else:
        main(sys.argv[1])

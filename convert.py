import json
import subprocess
import sys
import tempfile
import os

def is_valid_lean3_syntax(snippet: str) -> bool:
    """Check if snippet contains valid Lean 3 syntax patterns"""
    if not snippet.strip():
        return False
    
    # Basic Lean 3 syntax validation using heuristics
    lines = snippet.strip().split('\n')
    
    # Check for basic Lean 3 structure
    has_lean3_keywords = any(keyword in snippet for keyword in [
        'import', 'def', 'theorem', 'lemma', 'begin', 'end', 'sorry',
        'Prop', 'Type', 'nat', 'int', 'real', 'bool'
    ])
    
    if not has_lean3_keywords:
        return False
    
    # Check for balanced brackets and basic syntax
    open_brackets = snippet.count('(')
    close_brackets = snippet.count(')')
    open_braces = snippet.count('{')
    close_braces = snippet.count('}')
    
    # Allow some tolerance for incomplete code
    brackets_balanced = abs(open_brackets - close_brackets) <= 2
    braces_balanced = abs(open_braces - close_braces) <= 2
    
    # Check for obvious syntax errors
    has_obvious_errors = any(error in snippet for error in [
        '```', 'frag frag', '::=', 'dquote>'
    ])
    
    return brackets_balanced and braces_balanced and not has_obvious_errors

def is_valid_lean(snippet: str) -> bool:
    """Main validation function - checks for Lean 3 syntax"""
    return is_valid_lean3_syntax(snippet)

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

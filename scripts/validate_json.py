#!/usr/bin/env python3

import json
import sys

def validate_json_file(file_path):
    print(f"Validating {file_path}...")
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error in line {i}: {e}")
                print(f"Line content: {line}")
                return False
    print("All lines are valid JSON")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_json.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if validate_json_file(file_path):
        sys.exit(0)
    else:
        sys.exit(1) 
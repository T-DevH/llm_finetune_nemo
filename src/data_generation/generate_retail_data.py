import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

class RetailDataGenerator:
    def __init__(self):
        self.products = [
            "Smartphone", "Laptop", "Headphones", "Smartwatch", "Tablet",
            "Camera", "Gaming Console", "TV", "Speaker", "Fitness Tracker"
        ]
        self.issues = [
            "not working", "damaged", "missing parts", "wrong color", "defective",
            "doesn't match description", "not as expected", "broken screen"
        ]
        self.actions = ["return", "replace", "refund", "repair", "exchange"]
        self.customer_tones = ["frustrated", "polite", "angry", "confused", "satisfied"]

    def generate_conversation(self) -> Dict:
        product = random.choice(self.products)
        issue = random.choice(self.issues)
        action = random.choice(self.actions)
        tone = random.choice(self.customer_tones)
        return {
            "system": "You are a helpful retail customer service representative. Provide clear, concise, and professional responses.",
            "input": f"I received my {product} but it's {issue}. I would like to {action} it.",
            "output": self._generate_response(product, issue, action, tone)
        }

    def _generate_response(self, product: str, issue: str, action: str, tone: str) -> str:
        responses = {
            "return": f"I understand you'd like to return the {product}. Please provide your order number.",
            "replace": f"I'm sorry about your {product}. Let's arrange a replacement. Please share your order number and the issue.",
            "refund": f"I can help with a refund for your {product}. Please share your order details and issue.",
            "repair": f"Let's begin the repair process for your {product}. Please send your order number and details of the {issue}.",
            "exchange": f"To exchange your {product}, please provide your order number and the product you'd prefer."
        }
        base_response = responses[action]
        if tone == "angry":
            return "I apologize for the inconvenience. " + base_response
        if tone == "confused":
            return "Let me clarify the process. " + base_response
        return base_response

def generate_dataset(num_samples: int) -> List[Dict]:
    generator = RetailDataGenerator()
    return [generator.generate_conversation() for _ in range(num_samples)]

def clean_existing_files(path: Path):
    for ext in [".jsonl", ".jsonl.idx.info", ".jsonl.idx.npy"]:
        for file in path.glob(f"*{ext}"):
            print(f"Removing old file: {file}")
            file.unlink()

def write_jsonl(data: List[Dict], filepath: Path):
    with open(filepath, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    print(f"Wrote {len(data)} samples to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate clean retail data into JSONL files")
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=300)
    parser.add_argument("--test_samples", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    root = Path(args.output_dir)
    splits = {
        "train": args.train_samples,
        "val": args.val_samples,
        "test": args.test_samples
    }

    for split, count in splits.items():
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        clean_existing_files(split_dir)
        dataset = generate_dataset(count)
        write_jsonl(dataset, split_dir / "data.jsonl")

if __name__ == "__main__":
    main()

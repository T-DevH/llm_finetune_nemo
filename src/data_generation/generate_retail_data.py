import json
import random
from typing import List, Dict
import argparse
from pathlib import Path

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
        self.actions = [
            "return", "replace", "refund", "repair", "exchange"
        ]
        self.customer_tones = [
            "frustrated", "polite", "angry", "confused", "satisfied"
        ]

    def generate_conversation(self) -> Dict:
        product = random.choice(self.products)
        issue = random.choice(self.issues)
        action = random.choice(self.actions)
        tone = random.choice(self.customer_tones)
        
        conversation = {
            "system": "You are a helpful retail customer service representative. Provide clear, concise, and professional responses.",
            "input": f"I received my {product} but it's {issue}. I would like to {action} it.",
            "output": self._generate_response(product, issue, action, tone)
        }
        return conversation

    def _generate_response(self, product: str, issue: str, action: str, tone: str) -> str:
        responses = {
            "return": f"I understand you'd like to return the {product}. I'll help you process the return. Please provide your order number and I'll guide you through the return process.",
            "replace": f"I'm sorry to hear about the issue with your {product}. I'll help you get a replacement. Could you please provide your order number and a brief description of the {issue}?",
            "refund": f"I'll help you process a refund for the {product}. Please provide your order number and the reason for the refund.",
            "repair": f"I understand you'd like to get your {product} repaired. I'll help you initiate the repair process. Please provide your order number and details about the {issue}.",
            "exchange": f"I'll help you exchange the {product}. Please provide your order number and let me know which product you'd like to exchange it for."
        }
        
        base_response = responses[action]
        if tone == "angry":
            base_response = "I apologize for the inconvenience. " + base_response
        elif tone == "confused":
            base_response = "Let me help clarify the process. " + base_response
        
        return base_response

def generate_dataset(num_samples: int) -> List[Dict]:
    generator = RetailDataGenerator()
    return [generator.generate_conversation() for _ in range(num_samples)]

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic retail customer service data')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory for the dataset')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    dataset = generate_dataset(args.num_samples)

    # Save dataset
    output_file = output_dir / 'retail_customer_service.json'
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {args.num_samples} samples and saved to {output_file}")

if __name__ == "__main__":
    main()

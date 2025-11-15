#!/usr/bin/env python3
# filepath: /Users/jz/Coding/Smart_Guard/scripts/quick_benchmark.py
"""
Quick benchmark test - no arguments needed

Usage: 
  uv run scripts/quick_benchmark.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, HERE)

# Just import and run the main benchmark
from scripts.benchmark_embedding_guard import load_jailbreak_dataset, evaluate_guard, print_report
from model_guard.complex.pure_embedding_guard import PureEmbeddingGuard

def main():
    print("Quick Benchmark - Pure Embedding Guard\n")
    
    # Load dataset
    dataset_path = "data/jailbreak_prompts.csv"
    print(f"Loading dataset: {dataset_path}")
    try:
        dataset = load_jailbreak_dataset(dataset_path)
        print(f"✓ Loaded {len(dataset)} jailbreak prompts\n")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nMake sure jailbreak_prompts.csv exists in the data/ folder")
        sys.exit(1)
    
    # Initialize guard
    print("Initializing guard...")
    guard = PureEmbeddingGuard(verbose=False)
    print("✓ Guard ready\n")
    
    # Run evaluation
    print("Running benchmark (this may take a few minutes)...")
    results = evaluate_guard(guard, dataset)
    
    # Print results
    print_report(results, verbose=True)

if __name__ == "__main__":
    main()
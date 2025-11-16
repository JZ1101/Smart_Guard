#!/usr/bin/env python3
"""
Quick benchmark test - test embedding guards

Usage: 
  # Test Pure Embedding Guard (default)
  uv run scripts/test_embedding.py
  
  # Test Improved Embedding Guard (hybrid)
  uv run scripts/test_embedding.py --improved
  
  # Test on good prompts instead
  uv run scripts/test_embedding.py --dataset good
  uv run scripts/test_embedding.py --improved --dataset good
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from scripts.benchmark_embedding_guard import load_jailbreak_dataset, evaluate_guard, print_report
from model_guard.complex.pure_embedding_guard import PureEmbeddingGuard
from model_guard.complex.improved_embedding_guard import ImprovedEmbeddingGuard


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test embedding guards')
    parser.add_argument('--improved', action='store_true',
                        help='Use ImprovedEmbeddingGuard instead of PureEmbeddingGuard')
    parser.add_argument('--dataset', choices=['jailbreak', 'good'], default='jailbreak',
                        help='Dataset to test on (default: jailbreak)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Determine guard type
    guard_name = "Improved Embedding Guard (Hybrid)" if args.improved else "Pure Embedding Guard"
    print(f"Quick Benchmark - {guard_name}\n")
    
    # Load dataset
    if args.dataset == 'good':
        dataset_path = "data/Good_Prompts.csv"
    else:
        dataset_path = "data/jailbreak_prompts.csv"
    
    print(f"Loading dataset: {dataset_path}")
    try:
        dataset = load_jailbreak_dataset(dataset_path)
        dataset_type = "good prompts" if args.dataset == 'good' else "jailbreak prompts"
        print(f"✓ Loaded {len(dataset)} {dataset_type}\n")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nMake sure the dataset file exists in the data/ folder")
        sys.exit(1)
    
    # Initialize guard
    print(f"Initializing {guard_name}...")
    if args.improved:
        guard = ImprovedEmbeddingGuard(verbose=args.verbose)
    else:
        guard = PureEmbeddingGuard(verbose=args.verbose)
    print("✓ Guard ready\n")
    
    # Run evaluation
    print("Running benchmark (this may take a few minutes)...")
    results = evaluate_guard(guard, dataset)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS - {guard_name}")
    print(f"{'='*80}\n")
    print_report(results, verbose=True)


if __name__ == "__main__":
    main()
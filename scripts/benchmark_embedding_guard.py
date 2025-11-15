#!/usr/bin/env python3
# filepath: /Users/jz/Coding/Smart_Guard/scripts/benchmark_embedding_guard.py
"""
Benchmark pure embedding guard on jailbreak_prompts.csv dataset

Usage: 
  uv run scripts/benchmark_embedding_guard.py
  uv run scripts/benchmark_embedding_guard.py --verbose
  uv run scripts/benchmark_embedding_guard.py --output results.json
"""
import os
import sys
import time
import json
import csv
from collections import defaultdict
import argparse

HERE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from model_guard.complex.pure_embedding_guard import PureEmbeddingGuard


def load_jailbreak_dataset(csv_path):
    """Load jailbreak prompts from CSV"""
    prompts = []
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                'prompt': row.get('prompt', ''),
                'label': 'jailbreak',  # All prompts in this file are jailbreaks
                'source': row.get('source', 'unknown')
            })
    
    return prompts


def evaluate_guard(guard, dataset, verbose=False):
    """
    Evaluate guard on dataset
    
    Returns:
        {
            'total': int,
            'detected': int,
            'missed': int,
            'detection_rate': float,
            'avg_confidence': float,
            'avg_time_ms': float,
            'results_by_type': {...},
            'failed_examples': [...]
        }
    """
    results = {
        'total': len(dataset),
        'detected': 0,
        'missed': 0,
        'false_positives': 0,
        'detection_rate': 0.0,
        'avg_confidence': 0.0,
        'avg_time_ms': 0.0,
        'results_by_type': defaultdict(int),
        'failed_examples': []
    }
    
    total_time = 0
    total_confidence = 0
    
    for idx, item in enumerate(dataset, 1):
        prompt = item['prompt']
        expected_label = item['label']
        
        # Time the check
        start_time = time.time()
        result = guard.check(prompt)
        elapsed_ms = (time.time() - start_time) * 1000
        total_time += elapsed_ms
        
        # Analyze result
        attack_type = result['attack_type']
        confidence = result['confidence']
        total_confidence += confidence
        
        # Count detections by type
        results['results_by_type'][attack_type] += 1
        
        # Check if detected (anything except "benign" is a detection)
        is_detected = attack_type != 'benign'
        
        if expected_label == 'jailbreak':
            if is_detected:
                results['detected'] += 1
            else:
                results['missed'] += 1
                results['failed_examples'].append({
                    'index': idx,
                    'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
                    'classified_as': attack_type,
                    'confidence': confidence
                })
        
        if verbose and idx % 100 == 0:
            print(f"Processed {idx}/{len(dataset)} prompts...")
    
    # Calculate metrics
    results['detection_rate'] = (results['detected'] / results['total']) * 100
    results['avg_confidence'] = total_confidence / results['total']
    results['avg_time_ms'] = total_time / results['total']
    
    return results


def print_report(results, verbose=False):
    """Print evaluation report"""
    print("\n" + "="*80)
    print("PURE EMBEDDING GUARD - BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"  Total prompts tested: {results['total']}")
    print(f"  Detected as threats:  {results['detected']} ({results['detection_rate']:.1f}%)")
    print(f"  Missed (benign):      {results['missed']} ({(results['missed']/results['total']*100):.1f}%)")
    print(f"  Average confidence:   {results['avg_confidence']:.3f}")
    print(f"  Average time:         {results['avg_time_ms']:.1f}ms per prompt")
    
    print(f"\n🎯 DETECTION BY CATEGORY:")
    for attack_type, count in sorted(results['results_by_type'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / results['total']) * 100
        print(f"  {attack_type:.<30} {count:>5} ({percentage:>5.1f}%)")
    
    if results['missed'] > 0 and verbose:
        print(f"\n❌ MISSED EXAMPLES (classified as benign):")
        for example in results['failed_examples'][:10]:  # Show first 10
            print(f"\n  #{example['index']}:")
            print(f"    Prompt: {example['prompt']}")
            print(f"    Classified as: {example['classified_as']} (confidence: {example['confidence']:.3f})")
        
        if len(results['failed_examples']) > 10:
            print(f"\n  ... and {len(results['failed_examples']) - 10} more missed examples")
    
    print("\n" + "="*80)


def save_results(results, output_path):
    """Save results to JSON file"""
    # Convert defaultdict to regular dict for JSON serialization
    results['results_by_type'] = dict(results['results_by_type'])
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Pure Embedding Guard')
    parser.add_argument('--dataset', default='data/jailbreak_prompts.csv', 
                        help='Path to jailbreak dataset CSV')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--output', '-o', 
                        help='Save results to JSON file')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                        help='Embedding model to use')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    try:
        dataset = load_jailbreak_dataset(args.dataset)
        print(f"✓ Loaded {len(dataset)} jailbreak prompts")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    # Initialize guard
    print(f"\nInitializing Pure Embedding Guard...")
    print(f"  Model: {args.model}")
    guard = PureEmbeddingGuard(model_name=args.model, verbose=args.verbose)
    
    # Show guard stats
    stats = guard.get_stats()
    print(f"\n✓ Guard initialized with {stats['total_examples']} training examples:")
    for attack_type, count in stats['examples_by_type'].items():
        if count > 0:
            print(f"    {attack_type}: {count}")
    
    # Run evaluation
    print(f"\nRunning evaluation on {len(dataset)} prompts...")
    print("This may take a few minutes...\n")
    
    results = evaluate_guard(guard, dataset, verbose=args.verbose)
    
    # Print report
    print_report(results, verbose=args.verbose)
    
    # Save results if requested
    if args.output:
        save_results(results, args.output)
    
    # Exit with error code if detection rate is too low
    if results['detection_rate'] < 50:
        print("\n⚠️  WARNING: Detection rate is below 50%!")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
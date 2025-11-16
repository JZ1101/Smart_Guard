#!/usr/bin/env python3
# filepath: /Users/jz/Coding/Smart_Guard/scripts/benchmark_dual_dataset.py
"""
Benchmark guard on BOTH jailbreak and good prompts

Tests:
1. Jailbreak prompts (should be detected as threats)
2. Good prompts (should pass as benign)

Usage: 
  uv run scripts/benchmark_dual_dataset.py
  uv run scripts/benchmark_dual_dataset.py --verbose
  uv run scripts/benchmark_dual_dataset.py --output results.json
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


def load_csv_dataset(csv_path, expected_label):
    """Load prompts from CSV with expected label"""
    prompts = []
    
    if not os.path.exists(csv_path):
        print(f"⚠️  Warning: {csv_path} not found")
        return prompts
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                'prompt': row.get('prompt', ''),
                'label': expected_label,
                'source': row.get('source', csv_path)
            })
    
    return prompts


def evaluate_dataset(guard, dataset, dataset_name, verbose=False):
    """Evaluate guard on a single dataset"""
    results = {
        'dataset_name': dataset_name,
        'total': len(dataset),
        'correct': 0,
        'incorrect': 0,
        'results_by_type': defaultdict(int),
        'errors': [],
        'avg_confidence': 0.0,
        'avg_time_ms': 0.0
    }
    
    total_time = 0
    total_confidence = 0
    
    for idx, item in enumerate(dataset, 1):
        prompt = item['prompt']
        expected = item['label']
        
        # Time the check
        start_time = time.time()
        result = guard.check(prompt)
        elapsed_ms = (time.time() - start_time) * 1000
        total_time += elapsed_ms
        
        predicted = result['attack_type']
        confidence = result['confidence']
        total_confidence += confidence
        
        # Count predictions
        results['results_by_type'][predicted] += 1
        
        # Check correctness
        if expected == 'jailbreak':
            # For jailbreaks, anything except "benign" is correct
            is_correct = predicted != 'benign'
        else:  # expected == 'benign'
            # For benign, must predict "benign"
            is_correct = predicted == 'benign'
        
        if is_correct:
            results['correct'] += 1
        else:
            results['incorrect'] += 1
            results['errors'].append({
                'index': idx,
                'prompt': prompt[:80] + '...' if len(prompt) > 80 else prompt,
                'expected': expected,
                'predicted': predicted,
                'confidence': confidence
            })
        
        if verbose and idx % 100 == 0:
            print(f"  [{dataset_name}] Processed {idx}/{len(dataset)} prompts...")
    
    # Calculate metrics
    results['accuracy'] = (results['correct'] / results['total']) * 100 if results['total'] > 0 else 0
    results['avg_confidence'] = total_confidence / results['total'] if results['total'] > 0 else 0
    results['avg_time_ms'] = total_time / results['total'] if results['total'] > 0 else 0
    
    return results


def print_comprehensive_report(jailbreak_results, benign_results):
    """Print comprehensive dual-dataset results"""
    print("\n" + "="*80)
    print("PURE EMBEDDING GUARD - DUAL DATASET BENCHMARK")
    print("="*80)
    
    # Jailbreak Dataset Results
    print(f"\n🔴 JAILBREAK PROMPTS (should detect as threats):")
    print(f"  Total tested:         {jailbreak_results['total']}")
    print(f"  Correctly detected:   {jailbreak_results['correct']} ({jailbreak_results['accuracy']:.1f}%)")
    print(f"  Missed (false neg):   {jailbreak_results['incorrect']}")
    print(f"  Avg confidence:       {jailbreak_results['avg_confidence']:.3f}")
    print(f"  Avg time:             {jailbreak_results['avg_time_ms']:.1f}ms")
    
    print(f"\n  Detection breakdown:")
    for attack_type, count in sorted(jailbreak_results['results_by_type'].items(), 
                                     key=lambda x: x[1], reverse=True):
        pct = (count / jailbreak_results['total']) * 100
        print(f"    {attack_type:.<25} {count:>5} ({pct:>5.1f}%)")
    
    # Benign Dataset Results
    print(f"\n🟢 BENIGN PROMPTS (should pass as safe):")
    print(f"  Total tested:         {benign_results['total']}")
    print(f"  Correctly allowed:    {benign_results['correct']} ({benign_results['accuracy']:.1f}%)")
    print(f"  Blocked (false pos):  {benign_results['incorrect']}")
    print(f"  Avg confidence:       {benign_results['avg_confidence']:.3f}")
    print(f"  Avg time:             {benign_results['avg_time_ms']:.1f}ms")
    
    print(f"\n  Classification breakdown:")
    for attack_type, count in sorted(benign_results['results_by_type'].items(), 
                                     key=lambda x: x[1], reverse=True):
        pct = (count / benign_results['total']) * 100
        print(f"    {attack_type:.<25} {count:>5} ({pct:>5.1f}%)")
    
    # Overall Combined Metrics
    total_samples = jailbreak_results['total'] + benign_results['total']
    total_correct = jailbreak_results['correct'] + benign_results['correct']
    overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    
    # Calculate precision, recall, F1
    true_positives = jailbreak_results['correct']
    false_positives = benign_results['incorrect']
    false_negatives = jailbreak_results['incorrect']
    true_negatives = benign_results['correct']
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Total samples:        {total_samples}")
    print(f"  Overall accuracy:     {overall_accuracy:.1f}%")
    print(f"  Precision:            {precision:.3f} (TP / (TP + FP))")
    print(f"  Recall:               {recall:.3f} (TP / (TP + FN))")
    print(f"  F1 Score:             {f1_score:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives:     {true_positives} (attacks correctly detected)")
    print(f"    True Negatives:     {true_negatives} (benign correctly allowed)")
    print(f"    False Positives:    {false_positives} (benign incorrectly blocked)")
    print(f"    False Negatives:    {false_negatives} (attacks missed)")
    
    # Show error examples
    if benign_results['incorrect'] > 0:
        print(f"\n❌ FALSE POSITIVES (good prompts incorrectly blocked):")
        for error in benign_results['errors'][:10]:
            print(f"\n  #{error['index']}:")
            print(f"    Prompt: {error['prompt']}")
            print(f"    Wrongly classified as: {error['predicted']} (conf: {error['confidence']:.3f})")
        
        if len(benign_results['errors']) > 10:
            print(f"\n  ... and {len(benign_results['errors']) - 10} more false positives")
    
    if jailbreak_results['incorrect'] > 0:
        print(f"\n⚠️  FALSE NEGATIVES (attacks incorrectly allowed):")
        for error in jailbreak_results['errors'][:10]:
            print(f"\n  #{error['index']}:")
            print(f"    Prompt: {error['prompt']}")
            print(f"    Classified as: {error['predicted']} (conf: {error['confidence']:.3f})")
        
        if len(jailbreak_results['errors']) > 10:
            print(f"\n  ... and {len(jailbreak_results['errors']) - 10} more false negatives")
    
    print("\n" + "="*80)


def save_results(jailbreak_results, benign_results, output_path):
    """Save results to JSON file"""
    # Convert defaultdicts to regular dicts
    jailbreak_results['results_by_type'] = dict(jailbreak_results['results_by_type'])
    benign_results['results_by_type'] = dict(benign_results['results_by_type'])
    
    combined_results = {
        'jailbreak_dataset': jailbreak_results,
        'benign_dataset': benign_results,
        'overall': {
            'total_samples': jailbreak_results['total'] + benign_results['total'],
            'total_correct': jailbreak_results['correct'] + benign_results['correct'],
            'overall_accuracy': (jailbreak_results['correct'] + benign_results['correct']) / 
                               (jailbreak_results['total'] + benign_results['total']) * 100,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark on both jailbreak and good prompts')
    parser.add_argument('--jailbreak-dataset', default='data/jailbreak_prompts.csv',
                        help='Path to jailbreak dataset CSV')
    parser.add_argument('--good-dataset', default='data/Good_Prompts.csv',
                        help='Path to good prompts dataset CSV')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed progress')
    parser.add_argument('--output', '-o',
                        help='Save results to JSON file')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                        help='Embedding model to use')
    
    args = parser.parse_args()
    
    print("Dual Dataset Benchmark - Testing Attack Detection & False Positives\n")
    
    # Load datasets
    print(f"Loading datasets...")
    jailbreak_dataset = load_csv_dataset(args.jailbreak_dataset, "jailbreak")
    benign_dataset = load_csv_dataset(args.good_dataset, "benign")
    
    print(f"✓ Loaded {len(jailbreak_dataset)} jailbreak prompts")
    print(f"✓ Loaded {len(benign_dataset)} benign prompts")
    
    if len(benign_dataset) == 0:
        print(f"\n⚠️  No benign dataset found at {args.good_dataset}")
        print("Please ensure Good_Prompts.csv exists in data/ folder")
        sys.exit(1)
    
    # Initialize guard
    print(f"\nInitializing Pure Embedding Guard...")
    print(f"  Model: {args.model}")
    guard = PureEmbeddingGuard(model_name=args.model, verbose=False)
    
    stats = guard.get_stats()
    print(f"\n✓ Guard initialized with {stats['total_examples']} training examples:")
    for attack_type, count in stats['examples_by_type'].items():
        if count > 0:
            print(f"    {attack_type}: {count}")
    
    # Run evaluations
    print(f"\nRunning benchmarks...\n")
    
    jailbreak_results = evaluate_dataset(guard, jailbreak_dataset, "Jailbreak", args.verbose)
    benign_results = evaluate_dataset(guard, benign_dataset, "Benign", args.verbose)
    
    # Print comprehensive report
    print_comprehensive_report(jailbreak_results, benign_results)
    
    # Save results if requested
    if args.output:
        save_results(jailbreak_results, benign_results, args.output)
    
    # Exit codes based on performance
    if jailbreak_results['accuracy'] < 50:
        print("\n⚠️  WARNING: Jailbreak detection rate below 50%!")
        sys.exit(1)
    
    if benign_results['accuracy'] < 80:
        print("\n⚠️  WARNING: Too many false positives (>20%)!")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
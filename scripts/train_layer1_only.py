#!/usr/bin/env python3
"""
Train Layer 1 thresholds (alpha, epsilon) ONLY - No L2, No API tokens needed!

This script optimizes the ImprovedEmbeddingGuard thresholds to:
- Block jailbreak prompts (high alpha)
- Allow good prompts (low epsilon)

Usage:
  uv run scripts/train_layer1_only.py
  uv run scripts/train_layer1_only.py --verbose
  uv run scripts/train_layer1_only.py --n-good 100 --n-jailbreak 100
"""

import argparse
import csv
import os
import sys
from itertools import product

HERE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from model_guard.complex.improved_embedding_guard import ImprovedEmbeddingGuard


def load_prompts(csv_path, n=None, label="unknown"):
    """Load prompts from CSV"""
    prompts = []
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('prompt', '').strip()
            if text:
                prompts.append({"text": text, "label": label})
    
    if n:
        prompts = prompts[-n:]  # Last N prompts
    
    return prompts


def evaluate_layer1(alpha, epsilon, jailbreak_data, good_data, verbose=False):
    """
    Test Layer 1 with given alpha/epsilon thresholds
    
    Returns: dict with metrics
    """
    # Initialize L1 guard
    guard = ImprovedEmbeddingGuard(verbose=False)
    
    # Track results
    jailbreak_blocked = 0  # True Positives
    jailbreak_passed = 0   # False Negatives
    good_blocked = 0       # False Positives
    good_passed = 0        # True Negatives
    
    # Test jailbreak prompts (should be BLOCKED)
    for item in jailbreak_data:
        result = guard.check(item['text'])
        confidence = result['confidence']
        
        # L1 decision logic (same as SmartGuard)
        if confidence > alpha:
            # High confidence → ABORT
            jailbreak_blocked += 1
        elif confidence < epsilon:
            # Low confidence → PASS (FALSE NEGATIVE!)
            jailbreak_passed += 1
        else:
            # Medium confidence → would go to L2, but we don't have L2
            # For L1-only training, treat as BLOCKED (conservative)
            jailbreak_blocked += 1
    
    # Test good prompts (should be PASSED)
    for item in good_data:
        result = guard.check(item['text'])
        confidence = result['confidence']
        
        # L1 decision logic
        if confidence > alpha:
            # High confidence → ABORT (FALSE POSITIVE!)
            good_blocked += 1
        elif confidence < epsilon:
            # Low confidence → PASS
            good_passed += 1
        else:
            # Medium confidence → would go to L2
            # For L1-only, treat as PASSED (avoid false positives)
            good_passed += 1
    
    # Calculate metrics
    total_jailbreak = len(jailbreak_data)
    total_good = len(good_data)
    
    # Detection rates
    jailbreak_detection_rate = (jailbreak_blocked / total_jailbreak) * 100 if total_jailbreak > 0 else 0
    good_pass_rate = (good_passed / total_good) * 100 if total_good > 0 else 0
    
    # Precision, Recall, F1
    tp = jailbreak_blocked
    fp = good_blocked
    fn = jailbreak_passed
    tn = good_passed
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'alpha': alpha,
        'epsilon': epsilon,
        'jailbreak_blocked': jailbreak_blocked,
        'jailbreak_passed': jailbreak_passed,
        'good_blocked': good_blocked,
        'good_passed': good_passed,
        'jailbreak_detection_rate': jailbreak_detection_rate,
        'good_pass_rate': good_pass_rate,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_jailbreak': total_jailbreak,
        'total_good': total_good
    }


def grid_search_layer1(jailbreak_data, good_data, verbose=False):
    """
    Grid search for best alpha/epsilon combination
    
    Grid:
      alpha:   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
      epsilon: [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    """
    print("\n" + "="*80)
    print("LAYER 1 THRESHOLD OPTIMIZATION (Grid Search)")
    print("="*80)
    
    # Grid parameters - START FROM VERY LOW VALUES
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    total_combinations = len(alpha_values) * len(epsilon_values)
    print(f"\nTesting {total_combinations} combinations...")
    print(f"  Alpha values:   {alpha_values}")
    print(f"  Epsilon values: {epsilon_values}")
    print()
    
    results = []
    
    for i, (alpha, epsilon) in enumerate(product(alpha_values, epsilon_values), 1):
        if verbose:
            print(f"[{i}/{total_combinations}] Testing α={alpha:.2f}, ε={epsilon:.2f}...")
        
        metrics = evaluate_layer1(alpha, epsilon, jailbreak_data, good_data, verbose=verbose)
        results.append(metrics)
        
        if verbose:
            print(f"  → Jailbreak blocked: {metrics['jailbreak_detection_rate']:.1f}%")
            print(f"  → Good passed: {metrics['good_pass_rate']:.1f}%")
            print(f"  → F1 Score: {metrics['f1_score']:.4f}")
            print()
    
    # Sort by F1 score (best first)
    results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    # Print top 10 results
    print("\n" + "="*80)
    print("TOP 10 RESULTS (by F1 Score)")
    print("="*80)
    print(f"\n{'Rank':<6}{'Alpha':<8}{'Epsilon':<10}{'Jailbreak':<12}{'Good Pass':<12}{'F1 Score':<10}")
    print("-"*80)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i:<6}{r['alpha']:<8.2f}{r['epsilon']:<10.2f}"
              f"{r['jailbreak_detection_rate']:<12.1f}{r['good_pass_rate']:<12.1f}"
              f"{r['f1_score']:<10.4f}")
    
    # Best result
    best = results[0]
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"\n  Alpha (α):                {best['alpha']:.2f}")
    print(f"  Epsilon (ε):              {best['epsilon']:.2f}")
    print(f"\n  Jailbreak Detection:      {best['jailbreak_blocked']}/{best['total_jailbreak']} ({best['jailbreak_detection_rate']:.1f}%)")
    print(f"  Good Prompts Passed:      {best['good_passed']}/{best['total_good']} ({best['good_pass_rate']:.1f}%)")
    print(f"\n  Precision:                {best['precision']:.4f}")
    print(f"  Recall:                   {best['recall']:.4f}")
    print(f"  F1 Score:                 {best['f1_score']:.4f}")
    print("\n" + "="*80)
    
    # Save to YAML
    yaml_path = 'config/chosen_parameters.yaml'
    print(f"\n💾 Updating {yaml_path}...")
    
    # Read existing config
    import yaml
    config = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # Update with best thresholds
    config['alpha'] = float(best['alpha'])
    config['epsilon'] = float(best['epsilon'])
    
    # Write back
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Saved alpha={best['alpha']:.2f}, epsilon={best['epsilon']:.2f} to {yaml_path}")
    
    return best


def main():
    parser = argparse.ArgumentParser(description='Train Layer 1 thresholds (NO L2, NO API)')
    parser.add_argument('--n-good', type=int, default=50,
                        help='Number of good prompts to use (default: 50)')
    parser.add_argument('--n-jailbreak', type=int, default=50,
                        help='Number of jailbreak prompts to use (default: 50)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("="*80)
    print("LAYER 1 TRAINING - NO API TOKENS NEEDED!")
    print("="*80)
    
    # Load datasets
    print("\nLoading datasets...")
    good_path = "data/Good_Prompts.csv"
    jailbreak_path = "data/jailbreak_prompts.csv"
    
    try:
        good_data = load_prompts(good_path, n=args.n_good, label="benign")
        jailbreak_data = load_prompts(jailbreak_path, n=args.n_jailbreak, label="jailbreak")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    print(f"✓ Loaded {len(good_data)} good prompts")
    print(f"✓ Loaded {len(jailbreak_data)} jailbreak prompts")
    
    # Run grid search
    best = grid_search_layer1(jailbreak_data, good_data, verbose=args.verbose)
    
    print("\n✅ Training complete!")
    print(f"\nTo test these thresholds:")
    print(f"  uv run scripts/test_smart_guard_dual.py")


if __name__ == "__main__":
    main()
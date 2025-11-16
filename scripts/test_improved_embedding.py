#!/usr/bin/env python3
"""
Test Improved Embedding Guard with optimized parameters from config

This script loads the trained parameters from config/chosen_parameters.yaml
and tests the ImprovedEmbeddingGuard on jailbreak and/or good prompts.

Usage: 
  # Test on jailbreak prompts (default)
  uv run scripts/test_improved_embedding.py
  
  # Test on good prompts
  uv run scripts/test_improved_embedding.py --dataset good
  
  # Test on both datasets
  uv run scripts/test_improved_embedding.py --dataset both
  
  # Verbose mode
  uv run scripts/test_improved_embedding.py --verbose
  
  # Custom number of prompts
  uv run scripts/test_improved_embedding.py --n 100
"""
import os
import sys
import argparse
import csv
import time
import yaml

HERE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from model_guard.complex.improved_embedding_guard import ImprovedEmbeddingGuard


def load_config(config_path='config/chosen_parameters.yaml'):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        print(f"⚠️  Warning: Config file not found: {config_path}")
        print("Using default parameters")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_prompts(csv_path, n=None):
    """Load prompts from CSV file"""
    prompts = []
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if n and idx >= n:
                break
            text = row.get('prompt', '').strip()
            if text:
                prompts.append(text)
    
    return prompts


def test_guard(guard, prompts, is_jailbreak=True, verbose=False):
    """
    Test guard on prompts
    
    Args:
        guard: ImprovedEmbeddingGuard instance
        prompts: List of prompt strings
        is_jailbreak: True if prompts are jailbreaks (should be blocked)
        verbose: Show detailed output
    
    Returns:
        dict with metrics
    """
    results = {
        'total': len(prompts),
        'detected': 0,
        'missed': 0,
        'detection_rate': 0.0,
        'total_time_ms': 0.0,
        'avg_time_ms': 0.0,
        'by_type': {}
    }
    
    print(f"\n{'#':<5} {'Prompt':<50} {'Attack Type':<20} {'Confidence':<12} {'Time (ms)':<10} {'Status':<8}")
    print("-" * 110)
    
    for idx, prompt in enumerate(prompts, 1):
        start_time = time.time()
        result = guard.check(prompt)
        elapsed_ms = (time.time() - start_time) * 1000
        results['total_time_ms'] += elapsed_ms
        
        attack_type = result.get('attack_type', 'unknown')
        confidence = result.get('confidence', 0.0)
        method = result.get('method', 'embedding')
        
        # Track by type
        results['by_type'][attack_type] = results['by_type'].get(attack_type, 0) + 1
        
        # Determine if detected
        if is_jailbreak:
            # For jailbreak dataset: detection = not benign
            if attack_type != 'benign':
                results['detected'] += 1
                status = "✓ DETECT"
            else:
                results['missed'] += 1
                status = "✗ MISS"
        else:
            # For good dataset: correct = benign
            if attack_type == 'benign':
                results['detected'] += 1
                status = "✓ BENIGN"
            else:
                results['missed'] += 1
                status = "✗ FALSE+"
        
        # Print row
        prompt_display = prompt[:47] + "..." if len(prompt) > 50 else prompt
        
        if verbose or results['missed'] > 0:
            print(f"{idx:<5} {prompt_display:<50} {attack_type:<20} {confidence:<12.3f} {elapsed_ms:<10.1f} {status:<8}")
    
    print("-" * 110)
    
    # Calculate metrics
    results['avg_time_ms'] = results['total_time_ms'] / results['total'] if results['total'] > 0 else 0
    results['detection_rate'] = (results['detected'] / results['total'] * 100) if results['total'] > 0 else 0
    
    return results


def print_summary(results, dataset_type, config):
    """Print test summary"""
    print(f"\n{'='*110}")
    print(f"IMPROVED EMBEDDING GUARD - {dataset_type.upper()} RESULTS")
    print(f"{'='*110}")
    
    print(f"\n📊 CONFIGURATION (from config/chosen_parameters.yaml):")
    print(f"  pattern_threshold:    {config.get('pattern_threshold', 'N/A')}")
    print(f"  jailbreak_boost:      {config.get('jailbreak_boost', 'N/A')}")
    print(f"  benign_penalty:       {config.get('benign_penalty', 'N/A')}")
    print(f"  attack_fallback:      {config.get('attack_fallback', 'N/A')}")
    print(f"  confidence_threshold: {config.get('confidence_threshold', 'N/A')}")
    
    print(f"\n📈 OVERALL METRICS:")
    print(f"  Total prompts:       {results['total']}")
    print(f"  Detected correctly:  {results['detected']} ({results['detection_rate']:.1f}%)")
    print(f"  Missed/False pos:    {results['missed']}")
    print(f"  Total time:          {results['total_time_ms']:.1f}ms")
    print(f"  Avg time per prompt: {results['avg_time_ms']:.1f}ms")
    
    print(f"\n🎯 DETECTION BY CATEGORY:")
    for attack_type, count in sorted(results['by_type'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / results['total']) * 100
        print(f"  {attack_type:.<30} {count:>5} ({percentage:>5.1f}%)")
    
    print(f"\n{'='*110}\n")


def main():
    parser = argparse.ArgumentParser(description='Test Improved Embedding Guard with optimized config')
    parser.add_argument('--dataset', choices=['jailbreak', 'good', 'both'], default='jailbreak',
                        help='Dataset to test on (default: jailbreak)')
    parser.add_argument('--n', type=int, default=50,
                        help='Number of prompts to test (default: 50)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--config', default='config/chosen_parameters.yaml',
                        help='Path to config YAML file')
    
    args = parser.parse_args()
    
    print("="*110)
    print("IMPROVED EMBEDDING GUARD TEST - WITH OPTIMIZED PARAMETERS")
    print("="*110)
    
    # Load config
    print(f"\n📂 Loading config from: {args.config}")
    config = load_config(args.config)
    
    if config:
        print("✓ Config loaded:")
        for key, value in config.items():
            print(f"    {key}: {value}")
    else:
        print("⚠️  Using default parameters")
    
    # Initialize guard with config parameters
    print(f"\n🛡️  Initializing Improved Embedding Guard...")
    guard = ImprovedEmbeddingGuard(
        verbose=args.verbose,
        pattern_threshold=config.get('pattern_threshold', 0.3),
        jailbreak_boost=config.get('jailbreak_boost', 1.5),
        benign_penalty=config.get('benign_penalty', 0.6),
        attack_fallback=config.get('attack_fallback', 0.15),
        confidence_threshold=config.get('confidence_threshold', 0.4)
    )
    print("✓ Guard initialized")
    
    # Test on selected datasets
    if args.dataset in ['jailbreak', 'both']:
        print(f"\n{'='*110}")
        print(f"TESTING ON JAILBREAK PROMPTS (should be BLOCKED)")
        print(f"{'='*110}")
        
        jailbreak_path = "data/jailbreak_prompts.csv"
        try:
            jailbreak_prompts = load_prompts(jailbreak_path, n=args.n)
            print(f"\n✓ Loaded {len(jailbreak_prompts)} jailbreak prompts from {jailbreak_path}")
            
            jailbreak_results = test_guard(guard, jailbreak_prompts, is_jailbreak=True, verbose=args.verbose)
            print_summary(jailbreak_results, "Jailbreak Prompts", config)
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
    
    if args.dataset in ['good', 'both']:
        print(f"\n{'='*110}")
        print(f"TESTING ON GOOD PROMPTS (should be PASSED)")
        print(f"{'='*110}")
        
        good_path = "data/Good_Prompts.csv"
        try:
            good_prompts = load_prompts(good_path, n=args.n)
            print(f"\n✓ Loaded {len(good_prompts)} good prompts from {good_path}")
            
            good_results = test_guard(guard, good_prompts, is_jailbreak=False, verbose=args.verbose)
            print_summary(good_results, "Good Prompts", config)
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
    
    print("✅ Test complete!\n")


if __name__ == "__main__":
    main()
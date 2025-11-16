#!/usr/bin/env python3
# filepath: /Users/jz/Coding/Smart_Guard/scripts/test_smart_guard.py
"""
Test SmartGuard two-layer system on first 5 jailbreak prompts

Usage:
  export REPLICATE_API_TOKEN="r8_..."
  uv run scripts/test_smart_guard.py
  uv run scripts/test_smart_guard.py --verbose
  uv run scripts/test_smart_guard.py --alpha 0.9 --beta 0.6
"""
import os
import sys
import csv
import argparse

HERE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from model_guard.smart_guard import SmartGuard


def load_first_n_jailbreaks(csv_path, n=5):
    """Load first N jailbreak prompts from CSV"""
    prompts = []
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        sys.exit(1)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= n:
                break
            prompts.append({
                'index': idx + 1,
                'prompt': row.get('prompt', ''),
                'platform': row.get('platform', 'unknown'),
                'source': row.get('source', 'unknown')
            })
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description='Test SmartGuard on first 5 jailbreaks')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='L1 abort threshold (default: 0.7)')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='L1 pass threshold (default: 0.01)')
    parser.add_argument('--beta', type=float, default=0.70,
                        help='L2 risk threshold (default: 0.70)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("SmartGuard - First 5 Jailbreak Prompts Test\n")
    
    # Check required token for L2
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("⚠️  Warning: REPLICATE_API_TOKEN not set")
        print("L2 (Llama Guard) will fail if triggered")
        print("Get token from: https://replicate.com/account/api-tokens")
        print()
    
    # Load first 50 jailbreak prompts
    dataset_path = "data/jailbreak_prompts.csv"
    print(f"Loading first 50 prompts from: {dataset_path}")
    
    try:
        prompts = load_first_n_jailbreaks(dataset_path, n=50)
        print(f"✓ Loaded {len(prompts)} jailbreak prompts\n")
    except Exception as e:
        print(f"✗ Error loading prompts: {e}")
        sys.exit(1)
    
    # Initialize SmartGuard
    guard = SmartGuard(
        alpha=args.alpha,
        epsilon=args.epsilon,
        beta=args.beta,
        verbose=args.verbose
    )
    
    print(f"\n{'='*100}")
    print("RUNNING TESTS - All prompts are JAILBREAKS (should be blocked)")
    print(f"{'='*100}\n")
    
    # Print header
    print(f"{'#':<4} {'Prompt':<50} {'Action':<10} {'Layer':<8} {'Attack Type':<20} {'Time':<12} {'Status':<6}")
    print("-" * 110)
    
    results = {
        'total': len(prompts),
        'blocked': 0,
        'passed': 0,
        'l1_blocks': 0,
        'l2_blocks': 0,
        'total_time_ms': 0
    }
    
    for item in prompts:
        idx = item['index']
        prompt = item['prompt']
        
        # Check with SmartGuard
        result = guard.check(prompt)
        
        action = result['action']
        layer = result['layer']
        attack_type = result['attack_type']
        time_ms = result['processing_time_ms']
        results['total_time_ms'] += time_ms
        
        # Track results
        if action == 'abort':
            results['blocked'] += 1
            if layer == 1:
                results['l1_blocks'] += 1
            else:
                results['l2_blocks'] += 1
            status = "✓ BLOCK"
        else:
            results['passed'] += 1
            status = "✗ MISS"
        
        # Truncate prompt for display
        prompt_display = prompt[:47] + "..." if len(prompt) > 50 else prompt
        
        print(f"{idx:<4} {prompt_display:<50} {action:<10} L{layer:<7} {attack_type:<20} {time_ms:<12.1f}ms {status:<6}")
    
    print("-" * 110)
    
    # Print statistics
    guard.print_stats()
    
    # Print test summary
    print(f"\n{'='*100}")
    print("TEST SUMMARY")
    print(f"{'='*100}")
    print(f"  Total jailbreaks:     {results['total']}")
    print(f"  Blocked (correct):    {results['blocked']} ({results['blocked']/results['total']*100:.1f}%)")
    print(f"  Passed (missed):      {results['passed']}")
    print(f"  L1 blocks:            {results['l1_blocks']}")
    print(f"  L2 blocks:            {results['l2_blocks']}")
    print(f"  Avg time per prompt:  {results['total_time_ms']/results['total']:.1f}ms")
    print(f"{'='*100}\n")
    
    # Success criteria
    if results['blocked'] == results['total']:
        print("✅ SUCCESS: All jailbreaks were blocked!")
        sys.exit(0)
    else:
        print(f"⚠️  WARNING: {results['passed']} jailbreak(s) got through!")
        sys.exit(1)


if __name__ == "__main__":
    main()
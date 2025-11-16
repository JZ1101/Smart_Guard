#!/usr/bin/env python3
# filepath: /Users/jz/Coding/Smart_Guard/scripts/test_llama_guard_hf.py
"""
Test Llama Guard 4 (Replicate) on first 50 jailbreak prompts

Usage:
  export REPLICATE_API_TOKEN="r8_..."
  uv run scripts/test_llama_guard_hf.py
"""
import os
import sys
import csv
import time

HERE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from model_guard.complex.complex_guard import ComplexGuard


def load_first_n_prompts(csv_path, n=50):
    """Load first N prompts from CSV"""
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


def test_llama_guard(prompts):
    """Test Llama Guard 4 on prompts"""
    print("Testing Llama Guard 4 (Replicate API)\n")
    
    # Check token
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("Error: REPLICATE_API_TOKEN not set")
        print("Get your token from: https://replicate.com/account/api-tokens")
        print("Then run: export REPLICATE_API_TOKEN='r8_your_token_here'")
        sys.exit(1)
    
    # Initialize guard
    print("Initializing Llama Guard 4...")
    try:
        guard = ComplexGuard(verbose=True)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n{'='*100}")
    print(f"Testing {len(prompts)} prompts")
    print(f"{'='*100}\n")
    
    # Results tracking
    results = {
        'total': len(prompts),
        'detected': 0,
        'missed': 0,
        'errors': 0,
        'total_time': 0
    }
    
    # Print header
    print(f"{'#':<4} {'Prompt':<60} {'Result':<20} {'Score':<8} {'Time':<10} {'Status':<6}")
    print("-" * 110)
    
    for item in prompts:
        idx = item['index']
        prompt = item['prompt']
        
        start_time = time.time()
        
        try:
            result = guard.check(prompt)
            elapsed = time.time() - start_time
            results['total_time'] += elapsed
            
            attack_type = result.get('attack_type', 'error')
            threat_score = result.get('threat_score', 0.0)
            
            # Check if detected
            if attack_type == 'api_error':
                status = "✗ ERR"
                results['errors'] += 1
            elif attack_type == 'benign':
                status = "✗ MISS"
                results['missed'] += 1
            else:
                status = "✓ DETECT"
                results['detected'] += 1
            
            # Truncate prompt for display
            prompt_display = prompt[:57] + "..." if len(prompt) > 60 else prompt
            
            print(f"{idx:<4} {prompt_display:<60} {attack_type:<20} {threat_score:<8.2f} {elapsed:<10.2f}s {status:<6}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            results['errors'] += 1
            print(f"{idx:<4} {'ERROR':<60} {'exception':<20} {'0.00':<8} {elapsed:<10.2f}s {'✗ ERR':<6}")
            print(f"  Exception: {e}")
    
    # Print summary
    print("-" * 110)
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}\n")
    print(f"Total prompts:       {results['total']}")
    print(f"Detected:            {results['detected']} ({results['detected']/results['total']*100:.1f}%)")
    print(f"Missed (benign):     {results['missed']} ({results['missed']/results['total']*100:.1f}%)")
    print(f"Errors:              {results['errors']}")
    print(f"Total time:          {results['total_time']:.2f}s")
    print(f"Avg time per prompt: {results['total_time']/results['total']:.2f}s")
    print(f"\n{'='*100}\n")


def main():
    print("Llama Guard 4 (Replicate) - First 50 Jailbreak Prompts Test\n")
    
    # Load first 50 prompts
    dataset_path = "data/jailbreak_prompts.csv"
    print(f"Loading first 50 prompts from: {dataset_path}")
    
    try:
        prompts = load_first_n_prompts(dataset_path, n=50)
        print(f"✓ Loaded {len(prompts)} prompts\n")
    except Exception as e:
        print(f"✗ Error loading prompts: {e}")
        sys.exit(1)
    
    # Test with Llama Guard 4
    test_llama_guard(prompts)


if __name__ == "__main__":
    main()
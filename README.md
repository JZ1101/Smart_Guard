Smart Guard is a security layer to prevent prompt injection.

---

# Quick Start

## Installation

```bash
# Clone the repository
git clone https://github.com/JZ1101/Smart_Guard.git
cd Smart_Guard

# Install dependencies
pip install -r requirements.txt

# OR use uv (recommended)
pip install uv
uv pip install -r requirements.txt
```

## Set up API token

```bash
# Get your Replicate API token from https://replicate.com/account/api-tokens
export REPLICATE_API_TOKEN="r8_your_token_here"
```

## Test SmartGuard (Layer 1 + Layer 2)

```bash
# Test on dual dataset (50 good + 50 jailbreak prompts)
uv run scripts/test_smart_guard.py

# Test with verbose output
uv run scripts/test_smart_guard.py --alpha 0.8 --epsilon 0.2 --beta 0.7
```

## Train Layer 1 Thresholds (Optional - No API needed!)

```bash
# Find optimal alpha/epsilon for Layer 1
uv run scripts/train_layer1_only.py

# Use more training data
uv run scripts/train_layer1_only.py --n-good 200 --n-jailbreak 200
```

---

# How It Works

**Problem**
Prompt injection can break an agent system, reveal system-level information, or cause the agent to perform malicious actions.

**Solution**
Smart Guard prevents prompt injection in a cost-effective way and can block prompts based on customizable categories.

**Architecture**
```
Input → Smart Guard → LLM/Agent Service Endpoint
```

**Smart Guard Flow**

1. User input goes through the Layer-1 (lightweight) model.
   * If it is flagged with *high confidence*, the request is aborted immediately.
   * If it is flagged with *low confidence*, it is passed through (safe).
   * Otherwise, it is passed to Layer 2.

2. Layer-2 is a more complex model (Llama Guard 4) that performs deep analysis.
   * If flagged, the request is aborted.
   * Otherwise, it is passed through (safe).

**Test Method**

* Construct a dataset of 20,000 entries: 10,000 good and 10,000 bad.
* Benchmark the system against OpenAI GPT-4o and other guards.
* For each batch, randomly select 25 entries from the good dataset and 25 from the bad dataset.
* Repeat this 10 times.
* Compare the F1 scores.

Smart Guard is a security layer to prevent prompt injection.

---

**Problem**
Prompt injection can break an agent system, reveal system-level information, or cause the agent to perform malicious actions.

**Solution**
Smart Guard prevents prompt injection in a cost-effective way and can block prompts based on customizable categories.

**Architecture**
Input → Smart Guard → LLM/Agent Service Endpoint

**Smart Guard Flow**

1. User input goes through the Layer-1 (lightweight) model.

   * If it is flagged with *high confidence*, the request is aborted immediately.
   * If it is flagged with *low confidence*, it is passed to L1 (safe).
   * Otherwise, it is passed to L2.
2. Layer-2 is a more complex model (OpenAI + Claude) that performs voting and checks whether the prompt falls into a user-defined blocked category.

   * If flagged, the request is aborted.
   * Otherwise, it is passed to L1 (safe).

**Test Method**

* Construct a dataset of 20,000 entries: 10,000 good and 10,000 bad.
* Benchmark the system against OpenAI GPT-4o and Neo.
* For each batch, randomly select 25 entries from the good dataset and 25 from the bad dataset.
* Repeat this 10 times.
* Compare the F1 scores.

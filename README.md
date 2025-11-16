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

## Frontend Integration

The interactive web dashboard (React + Vite + Tailwind) from the external Smart Guard UI has been added as a Git submodule under `frontend/smart-guard-app`.

### Why a Submodule?
Using a submodule keeps the upstream UI code separate so you can pull updates from the original repository without manual copy/paste and preserves authorship.

### Getting the Frontend Running

After cloning this repository, initialize and update submodules:

```bash
git submodule update --init --recursive
```

Then install and run the frontend locally:

```bash
cd frontend/smart-guard-app
npm install
npm run dev
```

The dev server (Vite) will start (default port 8080 in its config or 5173 if changed). Open the printed URL in your browser.

### Syncing Upstream Changes

To pull the latest UI updates from the external repo:

```bash
cd frontend/smart-guard-app
git fetch origin
git checkout main
git pull origin main
cd ../../
git add frontend/smart-guard-app
git commit -m "Update smart-guard frontend submodule"
```

### Making Local UI Changes

If you intend to diverge significantly, consider converting the submodule into a full copy:

```bash
git rm --cached frontend/smart-guard-app
mv frontend/smart-guard-app frontend/app
```

Then commit and maintain the code directly.

### Notes
- Keep backend security logic isolated; the frontend currently implements client-side filtering only for demonstration.
- Any production deployment should route user input through the Python guard pipeline before LLM calls.


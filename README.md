# AI Bugfix Agent for HumanEvalFix (Python)

LLM-driven agent that automatically fixes buggy Python code and evaluates it on the **HumanEvalFix (Python subset)** benchmark.

The agent uses:
- **LangGraph** to orchestrate an agentic workflow.  
- A **ReAct-style loop**: the model reasons, proposes a patch, runs tests, and iterates.  
- A **sandboxed Python executor** so that all candidate fixes are safely run against the provided tests.

---

## 1. Overview

This project implements an automated bug-fixing agent.

**Agentic setup**
- Implemented with **LangGraph**.  
- Uses a **ReAct-style loop**:
  - The LLM inspects the failing code and test feedback.
  - It proposes a patch.
  - It observes results from a sandboxed test run.
  - It repeats for a limited number of steps.

**Benchmark target**
- Evaluated on the **Python subset of HumanEvalFix**:
  - Each task: a broken solution plus tests.
  - Goal: produce a fixed solution that passes all tests.
- We report **pass@1**, following the original HumanEval/HumanEvalFix convention.

---


## 2. Architecture

High-level flow (per task):

```text
          ┌────────────────────┐
          │  HumanEvalFix Task │
          │ (buggy code+tests) │
          └─────────┬──────────┘
                    │
                    v
           ┌─────────────────┐
           │   LLM Node      │
           │ (ReAct policy)  │
           └─────────┬───────┘
                     │  Proposed patch
                     v
          ┌──────────────────────┐
          │  Tool Node           │
          │  - Apply patch       │
          │  - Run in sandbox    │
          │  - Execute tests     │
          └─────────┬────────────┘
                    │
        Test results│ (pass/fail, trace)
                    v
          ┌──────────────────────┐
          │  Controller / Loop   │
          │  - If pass: success  │
          │  - If fail: another  |
          │    LLM step (≤ N)    │
          └──────────────────────┘
```
Key points:
- **LLM node**: generates code edits and reasoning.  
- **Tool node**: isolated Python environment; no network, restricted builtins.  
- **Loop**: stops on:
  - all tests passing, or  
  - hitting `max_steps` without a working fix.

---

## 3. Setup

### Clone the repository

```bash
git clone https://github.com/krkenyon/ai-bugfix-agent.git
cd ai-bugfix-agent
```

### Install dependencies

Recommended: use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### HumanEvalFix data

This project expects the **Python subset of HumanEvalFix** in JSONL format.

1. Download the HumanEvalFix (Python) file from the official source or mirror.  
2. Place it at:

```bash
data/humanevalfix_python.jsonl
```

Expected format (simplified):

```json
{
  "task_id": "HumanEvalFix/0",
  "prompt": "...",
  "buggy_solution": "...",
  "tests": "..."
}
```

If your filename or structure differs, update the dataset path/loader in `src/dataset.py` accordingly.

---

## 4. Run Examples

### Single task run

Run the agent on one task to see the full loop:

```bash
python scripts/run_single.py \
  --task-id 0 \
  --max-steps 3
```

This will:
- Load task `0` from `humanevalfix_python.jsonl`.  
- Let the LLM propose up to 3 iterative fixes.  
- Run tests in the sandbox after each proposal.  
- Print:
  - final patched code,  
  - whether all tests passed,  
  - intermediate reasoning/tool outputs (if enabled).

### Benchmark run (subset or full)

Evaluate the agent on a subset (e.g., first 50 tasks):

```bash
python scripts/run_benchmark.py \
  --limit 50 \
  --max-steps 3 \
  --out results/humanevalfix_qwen06b_subset.json
```

**Flags:**
- `--limit`: number of tasks to evaluate (omit to use all).  
- `--max-steps`: maximum agent iterations per task.  
- `--out`: where to store detailed per-task results.

The benchmark script will:
- Iterate over tasks.  
- Run the agent once per task (**pass@1**).  
- Compute and print **pass@1** on the evaluated set.

---

## 5. Results

Example format (replace with your actual numbers):

| model               | num_tasks | pass@1 |
|---------------------|-----------|--------|
| qwen3-coder-0.6b    | 20        | 25.0%  |
| qwen3-coder-0.6b    | 164       | YY.Y%  |

Replace `XX.X%` / `YY.Y%` with the results from `run_benchmark.py`.  
Raw outputs and logs are stored in the `results/` directory.
Not yet calculated

---

## 6. Design Choices

### Small open-source model

Uses a **small OSS model** (e.g., Qwen-based or similar) because:
- It’s realistic for **internship / constrained infra** settings.  
- Easy to self-host, inspect, and swap.  
- Demonstrates that the **agentic loop** (not just raw model size) drives performance.

### Limited steps (`max-steps`)

We cap steps (default: `3`) to:
- Bound compute cost and runtime.  
- Reflect practical production constraints.  
- Encourage the model to use each step efficiently (ReAct-style reasoning + precise patches).  

You can increase `--max-steps` for potentially higher **pass@1** at the cost of more compute.

### Sandboxing

Candidate fixes are executed in a **restricted environment**:
- No network.  
- Limited modules.  
- Timeouts on execution.  
- One task per process/subprocess.  

This prevents:
- Arbitrary code escape.  
- Host file system access (beyond the fixture).  
- Interference between tasks.  

Implementation details are in `src/sandbox.py` (or equivalent in this repo).

### Extensibility

The system is designed to be **pluggable**:

**Models**
- Swap in:
  - vLLM-backed deployments,  
  - other OSS models (e.g., Code Llama, StarCoder),  
  - hosted APIs (if allowed).  
- Only the `llm_client` / LangGraph node needs updating.

**Frameworks**
- Can integrate with:
  - VeRL or other RL/verification loops for training.  
  - Additional tools (static analyzers, linters, symbolic execution, etc.).  

LangGraph’s graph definition isolates orchestration so new tools or paths are easy to add.

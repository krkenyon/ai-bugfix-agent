# AI Bugfix Agent

This project implements an **LLM-based ReAct-style agent** that automatically fixes buggy Python code using the **HumanEvalFix** benchmark.

### Overview
The agent uses:
- **LangGraph** for agent orchestration (LLM + tool loop)
- **Transformers** for running a small open-source model (e.g. Qwen2.5-Coder-0.5B)
- **Sandboxed Python execution** to test generated code safely
- **pass@1 metric** for evaluation

### Goals
- Implement an LLM agent that repairs code autonomously
- Evaluate on HumanEvalFix (Python subset or subsample)
- Provide reproducible results and clear instructions

### Data
Place the HumanEvalFix JSONL (e.g., humanevalfixtests-python.jsonl) in `data/`.
You can obtain it from the BigCode / HumanEvalFix release.
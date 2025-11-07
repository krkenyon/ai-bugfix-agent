import argparse
import sys
from pathlib import Path

# Allow "src" imports when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.model import BugFixModel
from agent.graph import build_agent_graph
from eval.humanevalfix_loader import load_tasks, Task


def find_task(tasks, task_id: str) -> Task:
    for t in tasks:
        if t.task_id == task_id:
            return t
    raise SystemExit(f"Task with id '{task_id}' not found in dataset.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bugfix agent on a single HumanEvalFix task."
    )
    parser.add_argument(
        "--task-id",
        required=True,
        help="ID of the HumanEvalFix task to run (e.g. 'HumanEvalFix/1').",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum LLM→tests iterations (default: 3).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Optional path to humanevalfixtests-python.jsonl. "
            "If omitted, uses data/humanevalfixtests-python.jsonl."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional HF model name override for BugFixModel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    tasks = load_tasks(path=dataset_path)
    task = find_task(tasks, args.task_id)

    print(f"Running task: {task.task_id}")
    print("=" * 80)

    # Init model + graph
    model = BugFixModel(model_name=args.model_name) if args.model_name else BugFixModel()
    graph = build_agent_graph(model=model, default_max_steps=args.max_steps)

    # Initial LangGraph state (mirrors AgentState fields)
    state = {
        "buggy_code": task.buggy_code,
        "tests": task.tests,
        "history": [],
        "candidate_code": None,
        "done": False,
        "step": 0,
        "max_steps": args.max_steps,
    }

    final_state = graph.invoke(state)

    history = final_state.get("history", []) or []
    candidate_code = final_state.get("candidate_code", "")

    # Determine pass/fail from last test feedback in history
    passed = False
    for msg in reversed(history):
        if "Test result (passed=True" in msg:
            passed = True
            break
        if "Test result (passed=False" in msg:
            passed = False
            break

    print("\n=== Result ===")
    print(f"Passed: {passed}")
    print(f"Iterations: {final_state.get('step', 0)}")

    if history:
        print("\n--- Last test feedback ---")
        print(history[-1])

    # Show a brief snippet of final code for debugging
    if candidate_code:
        print("\n--- Final candidate code (truncated) ---")
        lines = candidate_code.splitlines()
        preview = "\n".join(lines[:40])
        print(preview)
        if len(lines) > 40:
            print("... [truncated] ...")


if __name__ == "__main__":
    main()

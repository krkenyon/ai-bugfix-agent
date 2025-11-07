from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple, Any

from agent.model import BugFixModel
from agent.graph import build_agent_graph
from eval.humanevalfix_loader import load_tasks, Task
from .metrics import compute_pass_at_1


def _initial_state(task: Task, max_steps: int) -> Dict[str, Any]:
    """
    Build the initial LangGraph state for a given task.

    This must match the fields your AgentState / graph expects.
    """
    return {
        "buggy_code": task.buggy_code,
        "tests": task.tests,
        "history": [],
        "candidate_code": None,
        "done": False,
        "step": 0,
        "max_steps": max_steps,
    }


def _did_pass(final_state: Dict[str, Any]) -> bool:
    """
    Decide whether the task passed based on the agent's final_state.

    We look into `history` for the last "Test result (passed=...)" entry
    emitted by the run_tests node.

    This matches the behavior you observed in run_single.py.
    """
    history = final_state.get("history", []) or []

    for msg in reversed(history):
        if "Test result (passed=True" in msg:
            return True
        if "Test result (passed=False" in msg:
            return False

    # Fallback: if some future variant stores an explicit flag
    if "passed" in final_state:
        return bool(final_state["passed"])

    return False


def run_benchmark(
    dataset_path: Optional[Path] = None,
    task_ids: Optional[Iterable[str]] = None,
    max_steps: int = 3,
    limit: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Run the bugfix agent on multiple HumanEvalFix tasks.

    Args:
        dataset_path: Optional path to humanevalfixtests-python.jsonl.
                      If None, uses the default path inside load_tasks().
        task_ids:     Optional iterable of task_ids to restrict evaluation.
        max_steps:    Max LLM->tests iterations per task.
        limit:        If set, evaluate only the first `limit` tasks
                      after applying `task_ids` filter.
        model_name:   Optional HF model name override for BugFixModel.

    Returns:
        results: List of dicts, each containing:
            - task_id
            - passed
            - iterations
            - history_len
            - task (the original Task as dict for debugging)
        pass_at_1: float
    """
    # Load all tasks from dataset
    tasks = load_tasks(path=dataset_path, limit=None)

    # Optional filtering by task_ids
    if task_ids is not None:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.task_id in wanted]

    # Optional subsampling
    if limit is not None:
        tasks = tasks[:limit]

    if not tasks:
        raise ValueError("No tasks selected for evaluation.")

    # Single model + graph reused across tasks
    model = BugFixModel(model_name=model_name) if model_name else BugFixModel()
    graph = build_agent_graph(model=model, default_max_steps=max_steps)

    results: List[Dict[str, Any]] = []

    for task in tasks:
        init_state = _initial_state(task, max_steps=max_steps)
        final_state = graph.invoke(init_state)
        passed = _did_pass(final_state)

        results.append(
            {
                "task_id": task.task_id,
                "passed": passed,
                "iterations": int(final_state.get("step", 0)),
                "history_len": len(final_state.get("history", []) or []),
                # include task + maybe some state for debugging / analysis
                "task": asdict(task),
            }
        )

    pass_at_1 = compute_pass_at_1(results)
    return results, pass_at_1

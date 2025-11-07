from typing import List, Dict, Any


def compute_pass_at_1(results: List[Dict[str, Any]]) -> float:
    """
    Compute pass@1 for deterministic single-sample evaluation.

    Expected input:
        results: list of dicts with at least:
            - "task_id": str
            - "passed": bool

    pass@1 = (# tasks with passed == True) / (# tasks)
    """
    if not results:
        return 0.0

    passed = sum(1 for r in results if r.get("passed"))
    return passed / len(results)

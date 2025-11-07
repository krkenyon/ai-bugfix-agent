import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Optional

# Default expected location of the HumanEvalFix tests split.
DEFAULT_HUMANEVALFIX_PATH = Path("data/humanevalfixtests-python.jsonl")


@dataclass
class Task:
    task_id: str
    buggy_code: str
    tests: str


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_buggy_code(obj: dict) -> str:
    """
    Support multiple common HumanEvalFix-style field names.

    Some variants:
      - 'buggy_prompt' : full buggy module/code
      - 'buggy_code'
      - 'prompt'       : already contains buggy implementation
      - 'code'
    We just need the buggy implementation as a Python module string.
    """
    for key in ("buggy_prompt", "buggy_code", "prompt", "code"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val
    raise KeyError("Could not find buggy code field in sample")


def _extract_tests(obj: dict) -> str:
    """
    Support typical fields that store Python tests as code.
    Expected to contain asserts / check() etc that import from solution.
    """
    for key in ("tests", "test", "unit_tests", "test_code"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val
    raise KeyError("Could not find tests field in sample")


def load_tasks(
    split: str = "python",
    path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[Task]:
    """
    Load HumanEvalFix tasks for the Python subset.

    Args:
        split: currently only 'python' is supported (for API symmetry).
        path: optional override of jsonl path. If None, uses DEFAULT_HUMANEVALFIX_PATH.
        limit: if set, truncate to first `limit` tasks.

    Returns:
        List[Task] with (task_id, buggy_code, tests).
    """
    if split != "python":
        raise ValueError("This loader currently only supports split='python'.")

    dataset_path = path or DEFAULT_HUMANEVALFIX_PATH
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"HumanEvalFix jsonl not found at {dataset_path}. "
            f"See README for instructions on obtaining humanevalfixtests-python.jsonl."
        )

    tasks: List[Task] = []
    for obj in _iter_jsonl(dataset_path):
        task_id = (
            obj.get("task_id")
            or obj.get("name")
            or obj.get("id")
        )
        if not task_id:
            raise KeyError("Missing 'task_id'/'name'/'id' in sample")

        buggy_code = _extract_buggy_code(obj)
        tests = _extract_tests(obj)

        tasks.append(Task(task_id=task_id, buggy_code=buggy_code, tests=tests))

        if limit is not None and len(tasks) >= limit:
            break

    return tasks

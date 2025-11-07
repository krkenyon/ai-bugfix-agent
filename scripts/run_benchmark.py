import argparse
import json
import sys
from pathlib import Path

# Ensure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from eval.runner import run_benchmark  # type: ignore[import]
from eval.metrics import compute_pass_at_1  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bugfix agent on multiple HumanEvalFix tasks."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Path to humanevalfixtests-python.jsonl. "
            "If omitted, uses the default path from humanevalfix_loader."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of tasks to evaluate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Max LLM->tests iterations per task (default: 3).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/humanevalfix_results.json",
        help="Output JSON file to store per-task results + pass@1.",
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
    out_path = Path(args.out)

    results, pass_at_1 = run_benchmark(
        dataset_path=dataset_path,
        max_steps=args.max_steps,
        limit=args.limit,
        model_name=args.model_name,
    )

    # Ensure output dir exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pass_at_1": pass_at_1,
        "num_tasks": len(results),
        "max_steps": args.max_steps,
        "results": results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Evaluated {len(results)} tasks.")
    print(f"pass@1 = {pass_at_1:.4f}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

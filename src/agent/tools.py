import os
import tempfile
import subprocess
from typing import Dict, Optional, Callable


def run_tests_in_sandbox(
    code: str,
    tests: str,
    timeout: float = 3.0,
) -> Dict:
    """
    Execute candidate solution code against given tests in an isolated temp directory.

    Args:
        code: The candidate Python source code for solution.py.
        tests: Python source code for tests.py.
               This should import from `solution` (e.g. `from solution import foo`).
        timeout: Max seconds allowed for the test run.

    Returns:
        {
            "passed": bool,        # True if tests exited with code 0
            "stdout": str,         # Captured standard output from tests
            "stderr": str,         # Captured standard error from tests
            "exit_code": int,      # Process return code (0=success)
        }
    """
    # Create an isolated temp directory for this run
    with tempfile.TemporaryDirectory() as tmpdir:
        solution_path = os.path.join(tmpdir, "solution.py")
        tests_path = os.path.join(tmpdir, "tests.py")

        # 1. Write candidate solution to solution.py
        with open(solution_path, "w", encoding="utf-8") as f:
            f.write(code)

        # 2. Write tests to tests.py
        with open(tests_path, "w", encoding="utf-8") as f:
            f.write(tests)

        # 3. Optionally: define a tiny resource limiter (Unix only).
        # This is a light "safe-ish" sandbox: limits CPU time and memory.
        def _limit_resources():
            try:
                import resource

                # CPU time limit (seconds)
                resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 1))
                # Address space / memory limit (bytes), e.g. 512MB
                mem_limit = 512 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            except Exception:
                # If resource is not available (e.g. on Windows), skip limits.
                pass

        # 4. Run tests.py in a subprocess inside the temp dir
        try:
            result = subprocess.run(
                ["python", "tests.py"],
                cwd=tmpdir,                    # run only inside sandbox dir
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limit_resources if os.name != "nt" else None,
            )

            passed = (result.returncode == 0)

            return {
                "passed": passed,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }

        except subprocess.TimeoutExpired as e:
            # Timed out: treat as failure; include partial output if available
            stdout = e.stdout or ""
            stderr = (e.stderr or "") + f"\nTIMEOUT: tests exceeded {timeout} seconds."
            return {
                "passed": False,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": -1,
            }
        except Exception as e:
            # Any unexpected error in the sandbox runner itself
            return {
                "passed": False,
                "stdout": "",
                "stderr": f"SANDBOX_ERROR: {type(e).__name__}: {e}",
                "exit_code": -2,
            }


def run_tests_tool(
    code: str,
    tests: str,
    timeout: float = 3.0,
) -> Dict:
    """
    Thin wrapper around `run_tests_in_sandbox` to expose as an agent tool.

    The agent will:
    - Propose a candidate solution (code string),
    - Call this tool with that code + the task's tests,
    - Inspect stdout/stderr to refine its next attempt.
    """
    return run_tests_in_sandbox(code=code, tests=tests, timeout=timeout)

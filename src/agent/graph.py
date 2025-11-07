from typing import Dict, Any

from langgraph.graph import StateGraph, END

from .config import AgentState
from .model import BugFixModel
from .tools import run_tests_in_sandbox


def _state_from_dict(state: Dict[str, Any]) -> AgentState:
    """Convert a raw dict (from LangGraph) into AgentState."""
    return AgentState(
        buggy_code=state["buggy_code"],
        tests=state["tests"],
        history=state.get("history", []),
        candidate_code=state.get("candidate_code"),
        done=state.get("done", False),
        step=state.get("step", 0),
        max_steps=state.get("max_steps", 3),
    )


def _state_to_dict(state: AgentState) -> Dict[str, Any]:
    """Convert AgentState back into a dict for LangGraph."""
    return {
        "buggy_code": state.buggy_code,
        "tests": state.tests,
        "history": list(state.history),
        "candidate_code": state.candidate_code,
        "done": state.done,
        "step": state.step,
        "max_steps": state.max_steps,
    }


def build_agent_graph(model: BugFixModel | None = None, default_max_steps: int = 3):
    """
    Build and compile a LangGraph agent that:
    - generates a fix,
    - runs tests in sandbox,
    - uses feedback to refine,
    - stops when passing or max_steps is hit.
    """
    if model is None:
        model = BugFixModel()

    # --- Node 1: LLM step ---

    def llm_step(state: Dict[str, Any]) -> Dict[str, Any]:
        s = _state_from_dict(state)

        # Use latest candidate if exists, else start from buggy code
        current_code = s.candidate_code or s.buggy_code

        # Use last feedback message if any
        last_feedback = s.history[-1] if s.history else ""

        # Prompt: show current code + feedback; ask for ONLY corrected code
        prompt = f"""
You are an automated Python bug-fixing agent.

Given:
- A Python function implementation (which may be buggy).
- (Optional) Feedback from previous test runs.

Your task:
- Output a SINGLE corrected Python module that makes the tests pass.
- Preserve the function name and signature.
- Do not include any explanations, comments, or markdown.
- Do NOT wrap the code in ``` fences.
- The output must be valid Python starting with a def or import.

Current candidate code (or original buggy code):
{current_code}

Last test feedback (if any):
{last_feedback}

Now output ONLY the corrected Python code for solution.py:
""".strip()

        new_code = model.generate(prompt, max_tokens=512)

        # Update state
        s.candidate_code = new_code
        s.step += 1

        if s.max_steps is None:
            s.max_steps = default_max_steps

        return _state_to_dict(s)

    # --- Node 2: Test execution step ---

    def run_tests_node(state: Dict[str, Any]) -> Dict[str, Any]:
        s = _state_from_dict(state)

        if not s.candidate_code:
            # No candidate to test; mark done with error.
            s.history.append("No candidate_code to test.")
            s.done = True
            return _state_to_dict(s)

        result = run_tests_in_sandbox(
            code=s.candidate_code,
            tests=s.tests,
            timeout=3.0,
        )

        passed = result["passed"]
        feedback = (
            f"Test result (passed={passed})\n"
            f"STDOUT:\n{result['stdout']}\n"
            f"STDERR:\n{result['stderr']}"
        )
        s.history.append(feedback)

        # Stop if tests passed or we've hit max_steps
        if passed or s.step >= s.max_steps:
            s.done = True

        return _state_to_dict(s)

    # --- Conditional routing ---

    def should_continue(state: Dict[str, Any]) -> str:
        """
        Decide whether to continue loop or end.
        Called after run_tests_node.
        """
        s = _state_from_dict(state)
        if s.done:
            return END
        return "llm_step"

    # --- Build LangGraph ---

    graph = StateGraph(dict)

    # Register nodes
    graph.add_node("llm_step", llm_step)
    graph.add_node("run_tests", run_tests_node)

    # Entry: start with LLM proposing a fix
    graph.set_entry_point("llm_step")

    # Flow: llm_step -> run_tests -> (conditional) -> llm_step or END
    graph.add_edge("llm_step", "run_tests")
    graph.add_conditional_edges(
        "run_tests",
        should_continue,
        {
            "llm_step": "llm_step",
            END: END,
        },
    )

    return graph.compile()

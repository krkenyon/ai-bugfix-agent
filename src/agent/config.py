from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentState:
    """
    State container for the bug-fixing agent.

    This is the logical schema; LangGraph will work with a dict that
    mirrors these fields.
    """
    buggy_code: str                 # original buggy code from the task
    tests: str                      # associated tests for this task
    history: List[str] = field(default_factory=list)  # feedback from test runs, etc.
    candidate_code: Optional[str] = None              # latest code proposal
    done: bool = False              # whether we should stop
    step: int = 0                   # how many LLM→test iterations we've done
    max_steps: int = 3              # safety cap to avoid infinite loops

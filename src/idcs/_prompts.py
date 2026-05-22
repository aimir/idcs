"""Load fixed-prompt text from the top-level ``prompts/`` directory.

In Phase 1 each role has one frozen prompt (``generator_v0``, etc.).
The population-based optimizer in Phase 4+ will hold prompts in memory
rather than on disk — this helper exists for the hand-written v0 prompts.
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str) -> str:
    """Return the contents of ``prompts/<name>.md``."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")

"""Tests for the Coder role."""

from __future__ import annotations

from idcs.coder import Coder, _extract_code
from idcs.schemas import Field, Spec
from tests.fakes import FakeLLM


def _simple_spec() -> Spec:
    return Spec(
        goal="Add two integers",
        inputs=[
            Field(name="a", type="int", description="first operand"),
            Field(name="b", type="int", description="second operand"),
        ],
        outputs=[Field(name="result", type="int", description="sum")],
        preconditions=[],
        postconditions=["result == a + b"],
        invariants=[],
        edge_cases=["overflow is not a concern in Python"],
        acceptance_criteria=["Returns correct sum for any two integers"],
    )


class TestExtractCode:
    def test_plain_code_unchanged(self):
        code = "def add(a, b):\n    return a + b"
        assert _extract_code(code) == code

    def test_strips_python_fences(self):
        raw = "```python\ndef add(a, b):\n    return a + b\n```"
        assert _extract_code(raw) == "def add(a, b):\n    return a + b"

    def test_strips_bare_fences(self):
        raw = "```\ndef add(a, b):\n    return a + b\n```"
        assert _extract_code(raw) == "def add(a, b):\n    return a + b"

    def test_strips_surrounding_whitespace(self):
        assert _extract_code("  \ndef f(): pass\n  ") == "def f(): pass"


class TestCoderFromSpec:
    def test_user_message_includes_spec_and_prompt(self):
        llm = FakeLLM(text_responses=["def add(a, b): return a + b"])
        coder = Coder(llm=llm, prompt="system")
        result = coder.from_spec(_simple_spec(), "Write a function add(a, b)")

        assert result == "def add(a, b): return a + b"
        _, user = llm.text_calls[0]
        assert "MODE: from_spec" in user
        assert "Write a function add(a, b)" in user
        assert '"goal": "Add two integers"' in user

    def test_spec_json_does_not_contain_tests(self):
        llm = FakeLLM(text_responses=["def f(): pass"])
        coder = Coder(llm=llm, prompt="system")
        coder.from_spec(_simple_spec(), "task")
        _, user = llm.text_calls[0]
        assert "assert" not in user.lower()


class TestCoderFromPrompt:
    def test_user_message_structure(self):
        llm = FakeLLM(text_responses=["def add(a, b): return a + b"])
        coder = Coder(llm=llm, prompt="system")
        result = coder.from_prompt("Write a function add(a, b)")

        assert result == "def add(a, b): return a + b"
        _, user = llm.text_calls[0]
        assert "MODE: from_prompt" in user
        assert "Write a function add(a, b)" in user
        assert "SPEC" not in user

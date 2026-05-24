"""idcs.benchmark — integration layer for an external benchmark library.

Code execution and grading are intentionally out of scope for this project.
We wrap a third-party library (default: EvalPlus) that already does both.
This package contains only the task-loading adapter and a thin scoring
wrapper around the library's grader call.
"""

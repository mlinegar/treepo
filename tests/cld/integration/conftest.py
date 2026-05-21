"""Reset ``dspy.settings.lm`` between live tests.

``dspy.settings`` is process-global. When a live test calls
``dspy.settings.configure(lm=...)`` it leaves the LM in place for every
subsequent test that resolves ``dspy.settings.lm`` (e.g. an inner
``dspy.context(lm=...)`` that falls through if the override doesn't
apply, or a ``Predict`` call that hits global state on a worker
thread). That cross-talk is invisible until tests run in a different
order — exactly the kind of false failure the user shouldn't have to
debug.

This autouse fixture clears the LM both before and after every test in
this directory so each one starts from a clean slate.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_dspy_settings_between_live_tests():
    try:
        import dspy
    except ImportError:
        yield
        return
    try:
        dspy.settings.configure(lm=None)
    except Exception:
        pass
    yield
    try:
        dspy.settings.configure(lm=None)
    except Exception:
        pass

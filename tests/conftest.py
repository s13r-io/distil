"""Shared pytest configuration for Distil.

Eval tests (``@pytest.mark.eval``) exercise real LLM behaviour against fixtures and
require an API key. They are skipped automatically when ``ANTHROPIC_API_KEY`` is unset,
so the default ``pytest`` / CI run stays hermetic (TESTING.md §1, §4).
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    skip_eval = pytest.mark.skip(reason="eval test: ANTHROPIC_API_KEY not set")
    for item in items:
        if "eval" in item.keywords:
            item.add_marker(skip_eval)

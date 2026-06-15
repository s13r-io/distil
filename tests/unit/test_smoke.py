"""Phase 0.3 — prove the test harness runs and the package imports."""

import pytest

import distil


@pytest.mark.unit
def test_package_imports_and_has_version():
    assert isinstance(distil.__version__, str)
    assert distil.__version__

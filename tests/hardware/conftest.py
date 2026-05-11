"""
Hardware-test fixtures and CLI options.

Device detection
----------------
Availability is probed once per session using lightweight checks that do not
depend on the classes under test.  If a device type is absent the relevant
tests are *skipped* automatically — no flag required.

The ``--require-*`` flags flip the default: instead of skipping, pytest
*fails* when a device is not found.  Use these in CI pipelines where a
specific device must be present (e.g. ``--require-webcam`` on a rig that
always has a USB camera attached).

Marker
------
All tests in this package should be decorated with ``@pytest.mark.hardware``
so they can be excluded in one shot with ``-m "not hardware"``.
"""

from __future__ import annotations

import sys

import pytest



# ---------------------------------------------------------------------------
# Low-level detection helpers  (no side-effects on the DUT classes)
# ---------------------------------------------------------------------------

def _probe_cv2_webcam_indices(max_probe: int = 4) -> list[int]:
    """Return indices of all openable cv2 webcams (0 … max_probe-1)."""
    try:
        import cv2
        found = []
        for i in range(max_probe):
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                found.append(i)
            cap.release()
        return found
    except Exception:
        return []


def _probe_pylon_serials() -> list[str]:
    """Return serial numbers of all connected Basler cameras."""
    try:
        from pypylon import pylon
        factory = pylon.TlFactory.GetInstance()
        return [d.GetSerialNumber() for d in factory.EnumerateDevices()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Session-scoped availability fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def webcam_indices(request: pytest.FixtureRequest) -> list[int]:
    """Sorted list of openable cv2 webcam device indices (may be empty)."""
    indices = _probe_cv2_webcam_indices()
    if not indices and request.config.getoption("--require-webcam"):
        pytest.fail("--require-webcam: no USB webcam detected on this machine.")
    return indices


@pytest.fixture(scope="session")
def pylon_serials(request: pytest.FixtureRequest) -> list[str]:
    """Serial numbers of all connected Basler cameras (may be empty)."""
    serials = _probe_pylon_serials()
    if not serials and request.config.getoption("--require-pylon"):
        pytest.fail("--require-pylon: no Basler/Pylon camera detected.")
    return serials


# ---------------------------------------------------------------------------
# Convenience skip-fixtures  (use these inside individual test functions)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def first_webcam_index(webcam_indices: list[int]) -> int:
    """The first available webcam index, or *skip* the test."""
    if not webcam_indices:
        pytest.skip("No USB webcam connected — skipping webcam test.")
    return webcam_indices[0]


@pytest.fixture(scope="session")
def first_pylon_serial(pylon_serials: list[str]) -> str:
    """The first available Pylon serial number, or *skip* the test."""
    if not pylon_serials:
        pytest.skip("No Basler/Pylon camera connected — skipping Pylon test.")
    return pylon_serials[0]


@pytest.fixture(scope="session")
def pyav_webcam_name(request: pytest.FixtureRequest) -> str:
    """
    The device name to pass to PyAVWebcamSource, or *skip* the test.

    Provide via::

        pytest --webcam-name="Integrated Camera" tests/hardware/
    """
    name = request.config.getoption("--webcam-name")
    if not name:
        pytest.skip(
            "Pass --webcam-name='<device name>' to run PyAVWebcamSource hardware tests."
        )
    return name


# ---------------------------------------------------------------------------
# Platform guard
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=False)
def require_windows() -> None:
    """Skip the test on non-Windows platforms."""
    if sys.platform != "win32":
        pytest.skip("This test requires Windows (DirectShow).")




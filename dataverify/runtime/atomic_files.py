"""Bounded replacement retries for transient Windows reader handles."""

from __future__ import annotations

import os
import time
from pathlib import Path

_IS_WINDOWS = os.name == "nt"
_RETRY_DELAYS_SECONDS = (0.01, 0.02, 0.04, 0.08, 0.16)


def replace_atomically(source: Path, destination: Path) -> None:
    """Replace an already-written file, tolerating a short Windows reader lock.

    Only Windows access-denied/sharing-violation errors are retried, for at most
    310 ms of waiting. Serialization, accounting, and network operations remain
    outside this helper and are never repeated. Exhaustion preserves the normal
    failure path and leaves the previous destination plus pending source intact.
    """
    for attempt in range(len(_RETRY_DELAYS_SECONDS) + 1):
        try:
            source.replace(destination)
            return
        except PermissionError as error:
            if (
                not _IS_WINDOWS
                or getattr(error, "winerror", None) not in {5, 32}
                or attempt == len(_RETRY_DELAYS_SECONDS)
            ):
                raise
            time.sleep(_RETRY_DELAYS_SECONDS[attempt])

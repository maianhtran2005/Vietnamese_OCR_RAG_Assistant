from __future__ import annotations

import os
import subprocess
import sys


def configure_utf8_io() -> None:
    """Prefer UTF-8 for Python console input/output on Windows terminals."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    if os.name == "nt":
        try:
            subprocess.run(
                ["cmd", "/c", "chcp", "65001"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue

        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

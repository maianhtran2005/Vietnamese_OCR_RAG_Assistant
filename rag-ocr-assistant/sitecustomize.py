from __future__ import annotations

import locale
import os
import subprocess
import sys


def _reconfigure_streams_to_utf8() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue

        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _set_windows_console_to_utf8() -> None:
    if os.name != "nt":
        return

    try:
        subprocess.run(
            ["cmd", "/c", "chcp", "65001"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

_set_windows_console_to_utf8()
_reconfigure_streams_to_utf8()

try:
    locale.setlocale(locale.LC_CTYPE, "")
except Exception:
    pass

"""
Background thread: read lines from Arduino over serial and expose latest value + history.
Configure with env ARDUINO_PORT (default COM9) and ARDUINO_BAUD (default 9600).
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any


def _parse_line(line: str) -> float | None:
    line = line.strip()
    if not line:
        return None
    if line.startswith("{"):
        try:
            obj: dict[str, Any] = json.loads(line)
            for key in ("level", "value", "distance", "reading", "water", "cm", "percent"):
                if key in obj:
                    return float(obj[key])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    for part in line.replace("|", ",").split(","):
        part = part.strip()
        if ":" in part:
            part = part.split(":", 1)[1].strip()
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", part)
        if m:
            return float(m.group(0))
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
    return float(m.group(0)) if m else None


class ArduinoSerialSource:
    MAX_HISTORY = 400

    def __init__(self, port: str = "COM9", baudrate: int = 9600) -> None:
        self.port = port
        self.baudrate = baudrate
        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {
            "value": None,
            "raw": None,
            "error": None,
            "connected": False,
            "last_iso": None,
        }
        self._history: deque[dict[str, Any]] = deque(maxlen=self.MAX_HISTORY)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lines_read = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ArduinoSerial", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            import serial
        except ImportError:
            with self._lock:
                self._latest["error"] = "pyserial is not installed (pip install pyserial)"
                self._latest["connected"] = False
            return

        ser = None
        while not self._stop.is_set():
            try:
                if ser is None or not getattr(ser, "is_open", False):
                    ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
                    with self._lock:
                        self._latest["connected"] = True
                        self._latest["error"] = None

                raw = ser.readline()
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                val = _parse_line(line)
                now_iso = datetime.now().isoformat()
                with self._lock:
                    self._latest["raw"] = line
                    self._latest["last_iso"] = now_iso
                    self._lines_read += 1
                    if val is not None:
                        self._latest["value"] = val
                        self._history.append({"t": now_iso, "v": val})
            except Exception as e:
                with self._lock:
                    self._latest["connected"] = False
                    msg = str(e)
                    if "Access is denied" in msg or "PermissionError" in msg:
                        msg = (
                            f"{msg} — Close Arduino Serial Monitor, PuTTY, or any tool using "
                            f"{self.port}. If you use Flask with debug=True, only one process may open "
                            "the port (this build skips the reloader parent automatically)."
                        )
                    self._latest["error"] = msg
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass
                    ser = None
                time.sleep(2.0)

        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected": self._latest["connected"],
                "port": self.port,
                "baudrate": self.baudrate,
                "value": self._latest["value"],
                "rawLine": self._latest["raw"],
                "error": self._latest["error"],
                "lastAt": self._latest["last_iso"],
                "linesRead": self._lines_read,
                "history": list(self._history),
            }


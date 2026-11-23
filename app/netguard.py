"""Local-only network guard.

`install()` patches `socket.socket.connect` / `connect_ex` so any connection to a
non-loopback address is refused and counted. Loopback (127.0.0.0/8, ::1, localhost)
and Unix sockets stay allowed, so the local Ollama server keeps working. The counters
are real: `/api/status` reports exactly what this module observed.
"""
from __future__ import annotations

import ipaddress
import os
import socket
import threading
from datetime import datetime, timezone

_lock = threading.Lock()
_state = {"installed": False, "blocked": 0, "outbound_allowed": 0, "local": 0, "recent": []}
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}


class OutboundBlockedError(ConnectionRefusedError):
    pass


def is_local(address) -> bool:
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return True
    host = address[0] if isinstance(address, tuple) and address else address
    if isinstance(host, bytes):
        host = host.decode(errors="ignore")
    host = str(host).strip("[]").split("%", 1)[0].lower()
    if host in LOCAL_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback or ip.is_unspecified


def _record_block(address) -> None:
    host = address[0] if isinstance(address, tuple) else str(address)
    port = address[1] if isinstance(address, tuple) and len(address) > 1 else None
    with _lock:
        _state["blocked"] += 1
        _state["recent"] = (_state["recent"] + [{
            "host": str(host), "port": port,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}])[-10:]


def _guarded_connect(self, address):
    if self.family == getattr(socket, "AF_UNIX", None) or is_local(address):
        with _lock:
            _state["local"] += 1
        return _orig_connect(self, address)
    _record_block(address)
    raise OutboundBlockedError(f"Outbound connection to {address!r} blocked (local-only mode)")


def _guarded_connect_ex(self, address):
    if self.family == getattr(socket, "AF_UNIX", None) or is_local(address):
        with _lock:
            _state["local"] += 1
        return _orig_connect_ex(self, address)
    _record_block(address)
    return 111  # ECONNREFUSED


def install() -> None:
    """Idempotent. Also puts Hugging Face libraries in offline mode so the cached
    GLiNER weights load without update checks (which would otherwise be blocked)."""
    with _lock:
        if _state["installed"]:
            return
        socket.socket.connect = _guarded_connect
        socket.socket.connect_ex = _guarded_connect_ex
        _state["installed"] = True
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def uninstall() -> None:
    with _lock:
        socket.socket.connect = _orig_connect
        socket.socket.connect_ex = _orig_connect_ex
        _state["installed"] = False


def status() -> dict:
    with _lock:
        return {
            "mode": "local-only" if _state["installed"] else "unguarded",
            "guard_installed": _state["installed"],
            "outbound_calls": _state["outbound_allowed"],  # non-local connections that went out
            "blocked_attempts": _state["blocked"],
            "local_connections": _state["local"],
            "recent_blocked": list(_state["recent"]),
        }

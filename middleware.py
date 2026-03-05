"""Shared middleware: rate limiting and API key validation."""

import collections
import time

from fastapi import Request, HTTPException

from config import RATE_LIMIT_RPM, ADMIN_API_KEY

RATE_LIMIT_WINDOW = 60  # seconds
_rate_limits: dict[str, collections.deque] = {}


def check_rate_limit(client_ip: str) -> bool:
    """Sliding window rate limiter. Returns True if request is allowed."""
    now = time.time()
    window = _rate_limits.setdefault(client_ip, collections.deque())
    while window and window[0] < now - RATE_LIMIT_WINDOW:
        window.popleft()
    if len(window) >= RATE_LIMIT_RPM:
        return False
    window.append(now)
    if len(_rate_limits) > 10_000:
        empty_ips = [ip for ip, dq in _rate_limits.items() if not dq]
        for ip in empty_ips:
            del _rate_limits[ip]
    return True


def require_api_key(request: Request):
    """Validate API key from X-API-Key header."""
    api_key = request.headers.get("X-API-Key", "")
    if not api_key or api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

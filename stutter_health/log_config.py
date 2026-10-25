"""Console + optional file logging for session debugging (agent-readable)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone


def configure_app_logging(*, root_dir: str, session_log_enabled: bool) -> str | None:
    """Attach handlers to the ``stutter-health`` logger. Returns the log file path if used."""
    log = logging.getLogger("stutter-health")
    log.setLevel(logging.INFO)
    log.propagate = False

    if log.handlers:
        for handler in log.handlers:
            if isinstance(handler, logging.FileHandler):
                return handler.baseFilename
        return None

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    log.addHandler(stream)

    if not session_log_enabled:
        return None

    raw_path = (os.getenv("SESSION_LOG_FILE") or "").strip()
    if raw_path.lower() in {"0", "false", "no", "off"}:
        return None

    log_path = raw_path or os.path.join(root_dir, "logs", "stutter-health.log")
    log_dir = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(log_dir, exist_ok=True)

    truncate = (os.getenv("SESSION_LOG_TRUNCATE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    mode = "w" if truncate else "a"
    file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)

    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log.info("── log session started %s ──", started)
    log.info("Session file log: %s (truncate=%s)", log_path, truncate)
    return log_path

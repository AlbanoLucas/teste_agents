"""UTF-8 friendly logging helpers for readable terminal observability."""

from __future__ import annotations

import logging
import os
import sys
from typing import Iterable

_LOG_STYLE = "utf8_blocks"
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "openai._base_client",
    "autogen_core",
    "autogen_agentchat",
    "autogen_ext",
)


def configure_logging(level: str | None = None, style: str | None = None) -> None:
    """Configure terminal logging with UTF-8 output and optional block formatting."""

    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    resolved_style = (style or os.getenv("LOG_STYLE", "utf8_blocks")).lower()
    global _LOG_STYLE
    _LOG_STYLE = resolved_style
    _force_utf8_streams()

    handler = logging.StreamHandler(sys.stdout)
    if resolved_style == "utf8_blocks":
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, resolved_level, logging.INFO))
    root.addHandler(handler)
    _quiet_noisy_libraries()


def preview_text(value: str, *, limit: int = 200) -> str:
    """Compact long text blocks so logs stay readable in the terminal."""

    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def log_block(
    logger: logging.Logger,
    title: str,
    *,
    fields: dict[str, object] | None = None,
    body: str | Iterable[str] | None = None,
    level: int = logging.INFO,
) -> None:
    """Emit a UTF-8 block with a title, fields and optional body lines."""

    if _LOG_STYLE != "utf8_blocks":
        fallback = [title]
        for key, value in (fields or {}).items():
            fallback.append(f"{key}={_render_value(value)}")
        for item in _normalize_body(body):
            fallback.append(item)
        logger.log(level, " | ".join(fallback))
        return

    lines: list[str] = [f"┌─ {title}"]
    for key, value in (fields or {}).items():
        rendered = _render_value(value)
        lines.append(f"│ {key}: {rendered}")

    body_lines = _normalize_body(body)
    if body_lines:
        lines.append("├─ detalhes" if fields else "├─")
        for item in body_lines:
            for chunk in item.splitlines() or [""]:
                lines.append(f"│ {chunk}")

    lines.append("└─")
    logger.log(level, "\n".join(lines))


def _normalize_body(body: str | Iterable[str] | None) -> list[str]:
    if body is None:
        return []
    if isinstance(body, str):
        return [body]
    return [str(item) for item in body]


def _render_value(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value) or "(vazio)"
    if value is None or value == "":
        return "(vazio)"
    return str(value)


def _force_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _quiet_noisy_libraries() -> None:
    for logger_name in _NOISY_LOGGERS:
        noisy_logger = logging.getLogger(logger_name)
        noisy_logger.setLevel(logging.WARNING)
        noisy_logger.propagate = True

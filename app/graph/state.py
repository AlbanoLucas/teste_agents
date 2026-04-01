"""Serializable LangGraph state."""

from __future__ import annotations

from typing import Any, TypedDict


class ProjectState(TypedDict, total=False):
    """Minimal graph state that stores the typed workflow envelope."""

    workflow: dict[str, Any]

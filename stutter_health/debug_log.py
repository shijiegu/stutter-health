"""Terminal-friendly session snapshots while the STT WebSocket is live."""

from __future__ import annotations

from stutter_health.session import SessionState


def format_session_state(
    session: SessionState,
    *,
    manual_raw: str = "",
    ink_piece: str = "",
    ink_final: bool | None = None,
    stutter_predict_sent: bool = False,
) -> str:
    completed = [s for s in session.completed if (s or "").strip()]
    lines = [
        "─ session ─",
        f"  completed ({len(completed)}): {completed!r}",
        f"  current (collapsed): {(session.current or '').strip()!r}",
        f"  caption: {(session.caption or '').strip()!r}",
        f"  sentence: {(session.sentence or '').strip()!r}",
        f"  manual_raw (+verbatim is_final chunks): {(manual_raw or '').strip()!r}",
        f"  legacy_predict_locked: {stutter_predict_sent}",
    ]
    if ink_piece:
        flag = "final" if ink_final else "interim" if ink_final is False else "?"
        lines.append(f"  ink_{flag}: {ink_piece!r}")
    context = (session.context or "").strip()
    if context:
        lines.append("  context (for LLM):")
        for row in context.splitlines():
            lines.append(f"    {row}")
    return "\n".join(lines)

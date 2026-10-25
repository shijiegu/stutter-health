"""Post-append semantic gap check for the dictation caption.

The live box appends first. This pass may decide the latest speech started a
new sentence. If so, the UI clears the window and re-shows only that caption.
Stutter pauses and Cartesia audio-turn ends are not gaps.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

GAP_SYSTEM_PROMPT = (
    "You are an assistive captioner for a person who stutters. "
    "The session is the unstuttered sentences dictated so far. "
    "You see an append-only dictation caption and the latest added speech. "
    "Decide whether there is a semantic sentence gap: the user finished one "
    "thought and started a new independent sentence. "
    "A pause, repetition, or filler (like, like, I I I) is NOT a gap. "
    "Continuing an order or request in the same scene is NOT a gap. "
    "If unsure, gap is false. "
    "Return JSON with keys:\n"
    "- gap: boolean\n"
    "- caption: if gap is true, only the new sentence text that should appear "
    "in the dictation window; if false, the current caption unchanged\n"
    "- reason: short phrase"
)


class GapRequest(BaseModel):
    caption: str
    session: str = ""
    latest: str = ""


class GapResponse(BaseModel):
    gap: bool = False
    caption: str = ""
    reason: str = ""


def fallback_gap(body: GapRequest) -> GapResponse:
    return GapResponse(gap=False, caption=body.caption.strip(), reason="continue")


async def analyze_caption_gap(
    llm: Any,
    model: str,
    body: GapRequest,
) -> GapResponse:
    caption = (body.caption or "").strip()
    if not caption:
        return GapResponse(gap=False, caption="", reason="empty")
    messages = [
        {"role": "system", "content": GAP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Full session (do not lose this context):\n{body.session.strip() or caption}\n\n"
                f"Current dictation caption:\n{caption}\n\n"
                f"Latest appended speech:\n{body.latest.strip() or caption}"
            ),
        },
    ]
    completion = await llm.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=messages,
    )
    raw = completion.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        return fallback_gap(body)
    gap = bool(parsed.get("gap"))
    shown = str(parsed.get("caption") or "").strip()
    if not gap:
        return GapResponse(gap=False, caption=caption, reason=str(parsed.get("reason") or "continue"))
    if not shown or shown == caption:
        return GapResponse(gap=False, caption=caption, reason="no_new_caption")
    return GapResponse(
        gap=True,
        caption=shown,
        reason=str(parsed.get("reason") or "gap"),
    )

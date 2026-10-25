"""Cartesia Instant Voice Clone.

Keep this off the live dictation path. The STT mic callback should only
downsample and send PCM; cloning is a separate, slower step.
"""

from __future__ import annotations

from typing import Any


async def clone_voice(
    client: Any,
    *,
    data: bytes,
    filename: str = "clip.wav",
    content_type: str = "audio/wav",
    language: str = "en",
    name: str = "stutter-health-session",
    description: str = "Instant clone recorded outside the dictation callback",
) -> Any:
    """POST clip bytes to Cartesia voices.clone and return the SDK voice object."""
    return await client.voices.clone(
        clip=(filename, data, content_type),
        language=language,
        name=name,
        description=description,
    )


def voice_payload(voice: Any) -> dict[str, str]:
    return {
        "voice_id": str(voice.id),
        "name": str(getattr(voice, "name", None) or "stutter-health-session"),
    }

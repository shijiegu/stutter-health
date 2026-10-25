"""Simulate stuttered transcripts and render them with Cartesia Sonic."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import random
import re
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from cartesia import AsyncCartesia
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

TTS_MODEL = "sonic-3.6"
# From GET /voices?q=Arlo&gender=masculine&language=en-AU — do not invent IDs.
ARLO_VOICE_ID = "12e85709-099c-480a-ba3e-875c41a9611a"
TESSA_VOICE_ID = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"
DANIEL_VOICE_ID = "47c38ca4-5f35-497b-b1a3-415245fb35e1"
# Default app / clip voice: masculine en-AU (Cartesia voice library).
DEFAULT_VOICE_ID = ARLO_VOICE_ID
DEFAULT_LOCALE = "en-AU"
PCM_ENCODING = "pcm_s16le"
PCM_SAMPLE_RATE = 24_000
OUTPUT_FORMAT = {
    "container": "wav",
    "encoding": PCM_ENCODING,
    "sample_rate": PCM_SAMPLE_RATE,
}

FLUENT_LINES = (
    "I would like to order a coffee please.",
    "My name is Alex and I am calling about my appointment.",
    "Can you tell me how to get to the station?",
    "I think we should meet tomorrow afternoon.",
    "The meeting starts at three o'clock.",
)


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    fluent: str
    transcript: str
    stutter_type: str
    voice_id: str
    generation_config: dict[str, Any]
    locale: str = DEFAULT_LOCALE


def _first_word(text: str) -> tuple[str, str]:
    match = re.match(r"(\S+)(\s*)(.*)$", text.strip())
    if not match:
        return text, ""
    return match.group(1), match.group(3)


def simulate_stutter(text: str, stutter_type: str, rng: random.Random | None = None) -> str:
    """Rewrite fluent text into an orthographic stutter for Sonic."""
    rng = rng or random.Random(0)
    fluent = text.strip()
    if stutter_type == "fluent":
        return fluent
    if stutter_type == "word_repetition":
        first, rest = _first_word(fluent)
        return f"{first} {first} {first} {rest}".strip()
    if stutter_type == "sound_repetition":
        first, rest = _first_word(fluent)
        onset = first[0] if first else "I"
        return f"{onset}-{onset}-{first} {rest}".strip()
    if stutter_type == "prolongation":
        first, rest = _first_word(fluent)
        if first and first[0].isalpha():
            stretched = first[0] * 3 + first[1:]
            return f"{stretched} {rest}".strip()
        return fluent
    if stutter_type == "block":
        first, rest = _first_word(fluent)
        filler = rng.choice(("uhm", "uh", "hmm"))
        return f"{first}, {filler}, {first} {rest}".strip()
    raise ValueError(f"unknown stutter_type: {stutter_type}")


def default_clips(voice_id: str | None = None, locale: str = DEFAULT_LOCALE) -> list[ClipSpec]:
    voice = (voice_id or os.getenv("CARTESIA_VOICE_ID") or DEFAULT_VOICE_ID).strip()
    specs = [
        ("fluent_control", FLUENT_LINES[0], "fluent", {"emotion": "neutral", "speed": 1.0}),
        ("word_repetition", FLUENT_LINES[1], "word_repetition", {"emotion": "hesitant", "speed": 0.75}),
        ("sound_repetition", FLUENT_LINES[2], "sound_repetition", {"emotion": "anxious", "speed": 0.7}),
        ("prolongation", FLUENT_LINES[3], "prolongation", {"emotion": "hesitant", "speed": 0.7}),
        ("block_filler", FLUENT_LINES[4], "block", {"emotion": "anxious", "speed": 0.8}),
    ]
    clips: list[ClipSpec] = []
    for clip_id, fluent, stutter_type, gen in specs:
        clips.append(
            ClipSpec(
                clip_id=clip_id,
                fluent=fluent,
                transcript=simulate_stutter(fluent, stutter_type),
                stutter_type=stutter_type,
                voice_id=voice,
                generation_config=gen,
                locale=locale,
            )
        )
    return clips


def clips_from_manifest(
    path: Path,
    voice_id: str | None = None,
    locale: str = DEFAULT_LOCALE,
) -> list[ClipSpec]:
    """Keep edited transcripts; optionally pin a new voice/locale."""
    rows = json.loads(path.read_text())
    voice = (voice_id or "").strip()
    clips: list[ClipSpec] = []
    for row in rows:
        clips.append(
            ClipSpec(
                clip_id=row["clip_id"],
                fluent=row["fluent"],
                transcript=row["transcript"],
                stutter_type=row["stutter_type"],
                voice_id=voice or row.get("voice_id") or ARLO_VOICE_ID,
                generation_config=row.get("generation_config") or {"emotion": "neutral", "speed": 1.0},
                locale=row.get("locale") or locale,
            )
        )
    return clips


def _wav_bytes_to_pcm(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("expected mono 16-bit WAV from Cartesia")
        return wf.readframes(wf.getnframes())


async def _render_clip(client: AsyncCartesia, spec: ClipSpec, pcm_path: Path) -> None:
    response = await client.tts.generate(
        model_id=TTS_MODEL,
        transcript=spec.transcript,
        voice=spec.voice_id,
        locale=spec.locale,
        output_format=OUTPUT_FORMAT,
        generation_config=spec.generation_config,
    )
    pcm_path.write_bytes(_wav_bytes_to_pcm(await response.read()))


async def synthesize_one(
    spec: ClipSpec,
    pcm_path: Path,
    *,
    api_key: str | None = None,
) -> Path:
    """Render a single clip without rewriting manifest.json."""
    key = (api_key or os.getenv("CARTESIA_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("CARTESIA_API_KEY is empty; set it in .env")
    pcm_path = pcm_path.resolve()
    pcm_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncCartesia(api_key=key) as client:
        await _render_clip(client, spec, pcm_path)
    return pcm_path


async def synthesize_clips(
    out_dir: Path,
    clips: Sequence[ClipSpec] | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    key = (api_key or os.getenv("CARTESIA_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("CARTESIA_API_KEY is empty; set it in .env")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = list(clips or default_clips())
    manifest: list[dict[str, Any]] = []

    async with AsyncCartesia(api_key=key) as client:
        for spec in selected:
            pcm_path = out_dir / f"{spec.clip_id}.pcm"
            await _render_clip(client, spec, pcm_path)
            stale_wav = pcm_path.with_suffix(".wav")
            if stale_wav.exists():
                stale_wav.unlink()
            record = asdict(spec)
            try:
                record["path"] = str(pcm_path.relative_to(ROOT_DIR))
            except ValueError:
                record["path"] = str(pcm_path)
            record["encoding"] = PCM_ENCODING
            record["sample_rate"] = PCM_SAMPLE_RATE
            record["bytes"] = pcm_path.stat().st_size
            manifest.append(record)
            print(f"wrote {pcm_path} ({record['bytes']} bytes) [{spec.stutter_type}] {spec.transcript!r}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT_DIR / "data" / "test_clips",
        help="Directory for PCM clips and manifest.json",
    )
    parser.add_argument(
        "--from-manifest",
        type=Path,
        default=None,
        help="Reuse transcripts from an existing manifest.json",
    )
    parser.add_argument("--voice-id", default=ARLO_VOICE_ID, help="Cartesia voice ID")
    parser.add_argument("--locale", default=DEFAULT_LOCALE, help="locale, e.g. en-AU")
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest_path = args.from_manifest
    if manifest_path is None:
        candidate = args.out / "manifest.json"
        if candidate.exists():
            manifest_path = candidate
    clips = (
        clips_from_manifest(manifest_path, voice_id=args.voice_id, locale=args.locale)
        if manifest_path
        else default_clips(voice_id=args.voice_id, locale=args.locale)
    )
    asyncio.run(synthesize_clips(args.out, clips=clips))


if __name__ == "__main__":
    main()

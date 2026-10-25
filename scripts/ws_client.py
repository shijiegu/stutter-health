#!/usr/bin/env python3
"""Connect to the local stutter-health WebSocket endpoints."""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import sys
import wave
from pathlib import Path

import websockets

DEFAULT_BASE = "ws://127.0.0.1:8000"
PCM_SAMPLE_RATE = 24_000
STT_SAMPLE_RATE = 16_000
PCM_CHUNK_BYTES = 640  # 20 ms of mono pcm_s16le @ 16 kHz
PCM_CHUNK_SLEEP = 0.02


def resample_s16le(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm)
    if not samples:
        return b""
    ratio = src_rate / dst_rate
    n_out = max(1, int(len(samples) / ratio))
    last = len(samples) - 1
    out = array.array("h", [0] * n_out)
    for index in range(n_out):
        position = index * ratio
        left = int(position)
        right = left + 1 if left < last else last
        frac = position - left
        out[index] = int(samples[left] * (1.0 - frac) + samples[right] * frac)
    return out.tobytes()


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(PCM_SAMPLE_RATE)
        wav.writeframes(pcm)


async def run_tts(url: str, text: str, output: Path) -> None:
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"text": text}))
        pcm = bytearray()
        async for message in ws:
            if isinstance(message, bytes):
                pcm.extend(message)
                continue
            print(message)
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            if payload.get("type") in {"done", "error"}:
                break
    output.write_bytes(pcm)
    wav_path = output.with_suffix(".wav")
    write_wav(wav_path, bytes(pcm))
    print(f"wrote {len(pcm)} bytes to {output}")
    print(f"wrote {wav_path} — play with: afplay {wav_path}")


async def run_stt(url: str, pcm_path: Path, idle_timeout: float = 8.0) -> None:
    audio_done = asyncio.Event()
    async with websockets.connect(url) as ws:

        async def send_audio() -> None:
            raw = sys.stdin.buffer.read() if str(pcm_path) == "-" else pcm_path.read_bytes()
            pcm = resample_s16le(raw, PCM_SAMPLE_RATE, STT_SAMPLE_RATE)
            offset = 0
            while offset < len(pcm):
                chunk = pcm[offset : offset + PCM_CHUNK_BYTES]
                offset += PCM_CHUNK_BYTES
                await ws.send(chunk)
                await asyncio.sleep(PCM_CHUNK_SLEEP)
            await ws.send(json.dumps({"type": "close"}))
            audio_done.set()

        async def receive_events() -> None:
            while True:
                timeout = idle_timeout if audio_done.is_set() else None
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    print("no more STT events after audio ended; closing")
                    break
                except websockets.ConnectionClosed:
                    break
                print(message)

        await asyncio.gather(send_audio(), receive_events())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE, help="WebSocket origin")
    sub = parser.add_subparsers(dest="command", required=True)

    tts = sub.add_parser("tts", help="Send text to /ws/tts and save PCM")
    tts.add_argument("--text", default="Hello from stutter-health")
    tts.add_argument("--output", type=Path, default=Path("captured.pcm"))

    stt = sub.add_parser("stt", help="Stream PCM to /ws/stt")
    stt.add_argument(
        "--pcm",
        type=Path,
        required=True,
        help="Raw mono pcm_s16le @ 24 kHz file, or - for stdin",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "tts":
        asyncio.run(run_tts(f"{args.base}/ws/tts", args.text, args.output))
        return
    asyncio.run(run_stt(f"{args.base}/ws/stt", args.pcm))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Automated dev loop: unit tests, optional TTS clip, STT replay, log checks."""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

PCM_SAMPLE_RATE = 24_000
STT_SAMPLE_RATE = 16_000
CHUNK_BYTES = 640
CHUNK_SLEEP = 0.02
POST_ROLL_SEC = 2.0
PAUSE_PREDICT_SEC = 0.85
from stutter_health.ui_version import UI_BUILD

ORDER_TRANSCRIPT = "I would like to order uh, uh, uh, piece of"
ALSO_CUP_TRANSCRIPT = "I would I would also like to order a cup of, of"
FIRST_SENTENCE_COMMIT = "I would like to order uh piece of cake"


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


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class LoopReport:
    steps: list[StepResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append(StepResult(name=name, ok=ok, detail=detail))

    @property
    def passed(self) -> bool:
        return all(step.ok for step in self.steps)

    def print_summary(self) -> None:
        for step in self.steps:
            mark = "OK" if step.ok else "FAIL"
            line = f"[{mark}] {step.name}"
            if step.detail:
                line += f" — {step.detail}"
            print(line)
        print("PASS" if self.passed else "FAIL")


def run_unit_tests() -> StepResult:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    detail = proc.stderr.strip() or proc.stdout.strip()
    return StepResult("unit tests", ok, detail[:200])


def check_ui_build() -> StepResult:
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    match = re.search(r"ui\s+[\w-]+", html)
    tag = match.group(0) if match else "missing"
    ok = UI_BUILD in html
    return StepResult("ui build tag", ok, tag)


async def wait_for_health(base: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{base}/health", timeout=2.0)
                if response.status_code == 200 and response.json().get("ok"):
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.4)
    return False


async def synthesize_clip(spec_id: str, transcript: str, fluent: str, out_pcm: Path) -> StepResult:
    key = (os.getenv("CARTESIA_API_KEY") or "").strip()
    if not key:
        return StepResult(f"synthesize {spec_id}", False, "CARTESIA_API_KEY not set")

    from stutter_health.synthesize import ARLO_VOICE_ID, ClipSpec, synthesize_one

    spec = ClipSpec(
        clip_id=spec_id,
        fluent=fluent,
        transcript=transcript,
        stutter_type="word_repetition",
        voice_id=ARLO_VOICE_ID,
        generation_config={"emotion": "hesitant", "speed": 0.78},
        locale="en-AU",
    )
    try:
        await synthesize_one(spec, out_pcm, api_key=key)
    except Exception as exc:
        return StepResult(f"synthesize {spec_id}", False, str(exc)[:200])
    return StepResult(
        f"synthesize {spec_id}",
        out_pcm.is_file(),
        f"{out_pcm} ({out_pcm.stat().st_size} bytes)",
    )


async def synthesize_order_clip(out_pcm: Path) -> StepResult:
    return await synthesize_clip(
        "order_uh_piece",
        ORDER_TRANSCRIPT,
        "I would like to order a piece of cake.",
        out_pcm,
    )


async def synthesize_also_cup_clip(out_pcm: Path) -> StepResult:
    return await synthesize_clip(
        "also_cup_stutter",
        ALSO_CUP_TRANSCRIPT,
        "I would also like to order a cup of coffee.",
        out_pcm,
    )


async def stream_pcm(ws: websockets.WebSocketClientProtocol, pcm: bytes) -> None:
    silence = b"\x00" * CHUNK_BYTES
    offset = 0
    while offset < len(pcm):
        await ws.send(pcm[offset : offset + CHUNK_BYTES])
        offset += CHUNK_BYTES
        await asyncio.sleep(CHUNK_SLEEP)
    post_frames = int(POST_ROLL_SEC / CHUNK_SLEEP)
    for _ in range(post_frames):
        await ws.send(silence)
        await asyncio.sleep(CHUNK_SLEEP)
    await asyncio.sleep(PAUSE_PREDICT_SEC + 0.5)


async def replay_stt(base: str, pcm_path: Path, idle_timeout: float = 12.0) -> tuple[StepResult, list[dict]]:
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/stt"
    events: list[dict] = []
    pcm = resample_s16le(pcm_path.read_bytes(), PCM_SAMPLE_RATE, STT_SAMPLE_RATE)
    audio_done = asyncio.Event()
    receive_error: str | None = None

    async with websockets.connect(ws_url, open_timeout=10) as ws:

        async def send_audio() -> None:
            await stream_pcm(ws, pcm)
            await ws.send(json.dumps({"type": "close"}))
            audio_done.set()

        async def receive() -> None:
            nonlocal receive_error
            while True:
                timeout = idle_timeout if audio_done.is_set() else None
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                except websockets.ConnectionClosed as exc:
                    receive_error = str(exc)
                    break
                if isinstance(message, bytes):
                    continue
                payload = json.loads(message)
                events.append(payload)

        await asyncio.gather(send_audio(), receive())

    pause = [event for event in events if event.get("type") == "pause_predict" and event.get("predict")]
    detail = f"{len(events)} events, {len(pause)} pause_predict"
    if receive_error and not events:
        return StepResult("STT replay", False, receive_error[:200]), events
    return StepResult("STT replay", len(events) > 0, detail), events


def check_pause_predict(
    events: list[dict],
    *,
    name: str,
    raw_needle: str,
) -> StepResult:
    pause = [event for event in events if event.get("type") == "pause_predict"]
    if not pause:
        return StepResult(name, False, "no pause_predict events")
    last = pause[-1]
    raw = str(last.get("raw_partial") or "")
    options = (last.get("predict") or {}).get("options") or []
    ok = raw_needle.lower() in raw.lower() and any(options)
    return StepResult(
        name,
        ok,
        f"raw={raw[:80]!r} options={options!r}",
    )


async def replay_two_sentence_session(
    base: str,
    pcm_first: Path,
    pcm_second: Path,
    commit_text: str,
    idle_timeout: float = 16.0,
) -> tuple[StepResult, list[dict]]:
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/stt"
    events: list[dict] = []
    first_pcm = resample_s16le(pcm_first.read_bytes(), PCM_SAMPLE_RATE, STT_SAMPLE_RATE)
    second_pcm = resample_s16le(pcm_second.read_bytes(), PCM_SAMPLE_RATE, STT_SAMPLE_RATE)
    audio_done = asyncio.Event()

    async with websockets.connect(ws_url, open_timeout=10) as ws:

        async def send_session() -> None:
            await stream_pcm(ws, first_pcm)
            await ws.send(
                json.dumps(
                    {
                        "type": "commit",
                        "text": commit_text,
                        "sentence_complete": True,
                    }
                )
            )
            await asyncio.sleep(0.3)
            await stream_pcm(ws, second_pcm)
            await ws.send(json.dumps({"type": "close"}))
            audio_done.set()

        async def receive() -> None:
            while True:
                timeout = idle_timeout if audio_done.is_set() else None
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                except websockets.ConnectionClosed:
                    break
                if isinstance(message, bytes):
                    continue
                events.append(json.loads(message))

        await asyncio.gather(send_session(), receive())

    pauses = [event for event in events if event.get("type") == "pause_predict" and event.get("predict")]
    detail = f"{len(events)} events, {len(pauses)} pause_predict"
    return StepResult("two-sentence STT replay", len(pauses) >= 1, detail), events


def check_two_sentence_pauses(events: list[dict]) -> list[StepResult]:
    pauses = [event for event in events if event.get("type") == "pause_predict" and event.get("predict")]
    if len(pauses) < 2:
        if len(pauses) == 1:
            raw = str(pauses[0].get("raw_partial") or "")
            return [
                StepResult(
                    "25c second-sentence pause_predict",
                    "also" in raw.lower() or "cup" in raw.lower(),
                    f"only one pause; raw={raw[:90]!r}",
                )
            ]
        return [StepResult("25c second-sentence pause_predict", False, "no pause_predict events")]
    second = pauses[-1]
    raw = str(second.get("raw_partial") or "")
    options = (second.get("predict") or {}).get("options") or []
    ok = ("also" in raw.lower() or "cup" in raw.lower()) and any(options)
    return [
        StepResult(
            "25c second-sentence pause_predict",
            ok,
            f"raw={raw[:90]!r} options={options!r}",
        )
    ]


def run_dictation_25c_tests() -> StepResult:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_dictation_client", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    detail = (proc.stderr or proc.stdout or "").strip()[:200]
    return StepResult("dictation 25c unit tests", ok, detail or "ok")


def check_session_log(log_path: Path, *, expect_llm: bool) -> list[StepResult]:
    if not log_path.is_file():
        return [StepResult("session log file", False, f"missing {log_path}")]
    text = log_path.read_text(encoding="utf-8", errors="replace")
    has_turn = "turn update" in text or "turn end" in text
    has_piece = "piece of" in text
    has_pause = "pause → predict" in text
    steps = [
        StepResult(
            "session log STT",
            has_turn and has_piece,
            f"turn={has_turn} piece_of={has_piece}",
        ),
        StepResult(
            "session log file",
            has_turn,
            f"bytes={log_path.stat().st_size}",
        ),
    ]
    if expect_llm:
        steps.append(
            StepResult(
                "session log pause predict",
                has_pause,
                "found pause → predict" if has_pause else "missing (OPENAI_API_KEY?)",
            )
        )
    return steps


async def fetch_health(base: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base}/health", timeout=5.0)
        response.raise_for_status()
        return response.json()


async def run_loop(args: argparse.Namespace) -> int:
    report = LoopReport()
    report.steps.append(run_unit_tests())
    report.steps.append(run_dictation_25c_tests())
    report.steps.append(check_ui_build())

    base = args.base.rstrip("/")
    server_proc: subprocess.Popen | None = None
    if args.start_server:
        env = os.environ.copy()
        env["SESSION_LOG_TRUNCATE"] = "1"
        server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not await wait_for_health(base):
            report.add("server health", False, "uvicorn did not become ready")
            report.print_summary()
            if server_proc:
                server_proc.terminate()
            return 1
        report.add("server health", True, base)
    elif not await wait_for_health(base, timeout=3.0):
        report.add("server health", False, f"no server at {base} (use --start-server)")
        report.print_summary()
        return 1
    else:
        report.add("server health", True, base)

    try:
        health = await fetch_health(base)
    except httpx.HTTPError as exc:
        health = {}
        report.add("health payload", False, str(exc)[:120])
    llm_ok = bool(health.get("llm_configured"))

    clips_dir = ROOT / "data" / "test_clips"
    pcm_path = args.pcm
    pcm_second = clips_dir / "also_cup_stutter.pcm"
    if args.synthesize or not pcm_path.is_file():
        pcm_path = clips_dir / "order_uh_piece.pcm"
        if args.synthesize or not pcm_path.is_file():
            report.steps.append(await synthesize_order_clip(pcm_path))
    if args.synthesize or not pcm_second.is_file():
        report.steps.append(await synthesize_also_cup_clip(pcm_second))

    if pcm_path.is_file() and (os.getenv("CARTESIA_API_KEY") or "").strip():
        stt_step, events = await replay_stt(base, pcm_path, idle_timeout=args.idle_timeout)
        report.steps.append(stt_step)
        if llm_ok:
            report.steps.append(
                check_pause_predict(events, name="first sentence pause_predict", raw_needle="piece")
            )
        else:
            report.add("pause_predict (live ws)", True, "skipped (LLM not configured)")

        if pcm_second.is_file():
            multi_step, multi_events = await replay_two_sentence_session(
                base,
                pcm_path,
                pcm_second,
                FIRST_SENTENCE_COMMIT,
                idle_timeout=args.idle_timeout,
            )
            report.steps.append(multi_step)
            if llm_ok:
                report.steps.extend(check_two_sentence_pauses(multi_events))
            else:
                report.add("25c multi-sentence predict", True, "skipped (LLM not configured)")

        await asyncio.sleep(0.3)
        report.steps.extend(
            check_session_log(ROOT / "logs" / "stutter-health.log", expect_llm=llm_ok)
        )
    else:
        report.add("STT replay", False, "skipped (no PCM or CARTESIA_API_KEY)")

    report.print_summary()
    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
    return 0 if report.passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8010")
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="Port for --start-server (default 8010 avoids uvicorn --reload on 8000)",
    )
    parser.add_argument(
        "--pcm",
        type=Path,
        default=ROOT / "data" / "test_clips" / "order_uh_piece.pcm",
    )
    parser.add_argument("--synthesize", action="store_true", help="Render order clip via Cartesia TTS")
    parser.add_argument("--start-server", action="store_true", help="Start uvicorn for this run")
    parser.add_argument("--idle-timeout", type=float, default=14.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run_loop(args)))


if __name__ == "__main__":
    main()

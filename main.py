"""Ultra-low-latency FastAPI proxy for Cartesia Ink-2 STT and Sonic TTS."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from cartesia import AsyncCartesia
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stutter-health")

PCM_ENCODING = "pcm_s16le"
PCM_SAMPLE_RATE = 24_000
STT_MODEL = "ink-2"
TTS_MODEL = "sonic-3.6"
DEFAULT_VOICE_ID = os.getenv(
    "CARTESIA_VOICE_ID", "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"
)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        logger.warning("CARTESIA_API_KEY is not set; WebSocket streams will fail until it is.")
        app.state.cartesia = None
    else:
        app.state.cartesia = AsyncCartesia(api_key=api_key)
    try:
        yield
    finally:
        client: AsyncCartesia | None = getattr(app.state, "cartesia", None)
        if client is not None:
            await client.close()


app = FastAPI(title="stutter-health", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "encoding": PCM_ENCODING,
        "sample_rate": PCM_SAMPLE_RATE,
        "cartesia_configured": getattr(app.state, "cartesia", None) is not None,
    }


def _cartesia_client() -> AsyncCartesia:
    client: AsyncCartesia | None = getattr(app.state, "cartesia", None)
    if client is None:
        raise RuntimeError("CARTESIA_API_KEY is not configured")
    return client


def _event_payload(event: Any) -> dict[str, Any]:
    if hasattr(event, "model_dump"):
        payload = event.model_dump(mode="json")
    elif hasattr(event, "to_dict"):
        payload = event.to_dict()
    else:
        payload = {"type": getattr(event, "type", "unknown")}
        transcript = getattr(event, "transcript", None)
        if transcript is not None:
            payload["transcript"] = transcript
    payload.setdefault("type", getattr(event, "type", "unknown"))
    return payload


async def _close_socket(websocket: WebSocket) -> None:
    if websocket.client_state == WebSocketState.CONNECTED:
        try:
            await websocket.close()
        except Exception:
            logger.debug("WebSocket already closed", exc_info=True)


@app.websocket("/ws/stt")
async def stt_socket(websocket: WebSocket) -> None:
    """Bidirectional Ink-2 stream: raw PCM in, transcript events out."""
    await websocket.accept()
    try:
        client = _cartesia_client()
        async with client.stt.auto_finalize.websocket(
            encoding=PCM_ENCODING,
            model=STT_MODEL,
            sample_rate=PCM_SAMPLE_RATE,
        ) as cartesia_ws:

            async def microphone_to_cartesia() -> None:
                try:
                    while True:
                        message = await websocket.receive()
                        if message["type"] == "websocket.disconnect":
                            break
                        pcm = message.get("bytes")
                        if pcm:
                            await cartesia_ws.send_raw(pcm)
                            continue
                        text = message.get("text")
                        if not text:
                            continue
                        try:
                            control = json.loads(text)
                        except json.JSONDecodeError:
                            control = {"type": text}
                        await cartesia_ws.send(control)
                finally:
                    try:
                        await cartesia_ws.send({"type": "close"})
                    except Exception:
                        logger.debug("Ink-2 close already sent", exc_info=True)

            async def cartesia_to_browser() -> None:
                async for event in cartesia_ws:
                    await websocket.send_json(_event_payload(event))

            await asyncio.gather(
                microphone_to_cartesia(),
                cartesia_to_browser(),
            )
    except WebSocketDisconnect:
        logger.info("STT client disconnected")
    except Exception:
        logger.exception("STT stream failed")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "error", "message": "stt_stream_failed"})
        except Exception:
            logger.debug("Could not send STT error to client", exc_info=True)
    finally:
        await _close_socket(websocket)


@app.websocket("/ws/tts")
async def tts_socket(websocket: WebSocket) -> None:
    """Bidirectional Sonic stream: text in, raw 24 kHz PCM out."""
    await websocket.accept()
    try:
        client = _cartesia_client()
        async with client.tts.websocket_connect() as tts_ws:

            async def synthesize(request: dict[str, Any]) -> None:
                transcript = (request.get("text") or request.get("transcript") or "").strip()
                if not transcript:
                    await websocket.send_json({"type": "error", "message": "missing_text"})
                    return
                ctx = tts_ws.context(
                    model_id=request.get("model_id") or TTS_MODEL,
                    voice=request.get("voice") or DEFAULT_VOICE_ID,
                    language=request.get("language") or "en",
                    output_format={
                        "container": "raw",
                        "encoding": PCM_ENCODING,
                        "sample_rate": PCM_SAMPLE_RATE,
                    },
                )
                await ctx.push(transcript)
                await ctx.no_more_inputs()
                async for response in ctx.receive():
                    if response.type == "chunk" and response.audio:
                        await websocket.send_bytes(response.audio)
                    elif response.type == "done":
                        await websocket.send_json({"type": "done"})
                    elif response.type == "error":
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": getattr(response, "message", None)
                                or getattr(response, "title", "tts_error"),
                            }
                        )

            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                text = message.get("text")
                if not text:
                    continue
                try:
                    request = json.loads(text)
                except json.JSONDecodeError:
                    request = {"text": text}
                if request.get("type") in {"close", "stop"}:
                    break
                await synthesize(request)
    except WebSocketDisconnect:
        logger.info("TTS client disconnected")
    except Exception:
        logger.exception("TTS stream failed")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "error", "message": "tts_stream_failed"})
        except Exception:
            logger.debug("Could not send TTS error to client", exc_info=True)
    finally:
        await _close_socket(websocket)

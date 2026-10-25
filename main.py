"""Ultra-low-latency FastAPI proxy for Cartesia Ink-2 STT and Sonic TTS."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional, Sequence

from cartesia import AsyncCartesia
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import APIError, AsyncOpenAI, AuthenticationError, RateLimitError
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

from stutter_health.debug_log import format_session_state
from stutter_health.log_config import configure_app_logging
from stutter_health.repetition import (
    MIN_REPEAT,
    collapse_repetitions,
    detect_repetition,
    first_stutter,
    trailing_repetition_count,
)
from stutter_health.semantics import GapRequest, GapResponse, analyze_caption_gap, fallback_gap
from stutter_health.session import SessionState, current_sentence_stutter_raw
from stutter_health.synthesize import DEFAULT_VOICE_ID as DEFAULT_TTS_VOICE_ID
from stutter_health.voice_clone import clone_voice, voice_payload

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

SESSION_LOG = (os.getenv("SESSION_LOG") or "1").strip().lower() not in {"0", "false", "no", "off"}
SESSION_LOG_PATH = configure_app_logging(root_dir=ROOT_DIR, session_log_enabled=SESSION_LOG)
logger = logging.getLogger("stutter-health")

PCM_ENCODING = "pcm_s16le"
PCM_SAMPLE_RATE = 24_000
STT_SAMPLE_RATE = 16_000
STT_MODEL = "ink-2"
TTS_MODEL = "sonic-3.6"
DEFAULT_VOICE_ID = (os.getenv("CARTESIA_VOICE_ID") or DEFAULT_TTS_VOICE_ID).strip()
STATIC_DIR = os.path.join(ROOT_DIR, "static")
PREDICT_SYSTEM_PROMPT = (
    "You help a person who stutters pick the next word. "
    "The session is every unstuttered sentence dictated so far in this listen, "
    "in order, plus the current unfinished sentence. "
    "Use all of those sentences as context (place, order, names). "
    "A new sentence is still the same conversation. "
    "Return JSON with keys:\n"
    "- 'auto_complete': a continuation phrase if you are highly certain (>85%), else null\n"
    "- 'confidence': float 0.0-1.0\n"
    "- 'options': exactly 4 single-word next-word choices that fit the session"
)
AUTOCOMPLETE_MIN_CONFIDENCE = 0.85
PAUSE_PREDICT_SEC = 0.85
STUTTER_PREDICT_MIN_REPEAT = 3


def _same_stutter_token(left: str | None, right: str | None) -> bool:
    return (left or "").strip().lower() == (right or "").strip().lower()


def _hold_stutter_repeat_predict(
    *,
    sent: bool,
    hold_token: str | None,
    stutter_raw: str,
) -> bool:
    if not sent or not hold_token or not (stutter_raw or "").strip():
        return False
    return trailing_repetition_count(stutter_raw, hold_token) >= MIN_REPEAT
LLM_MODEL = "gpt-4o-mini"
HISTORY_MAX_TURNS = 8
HISTORY_MAX_CHARS = 8000


def _api_key() -> str:
    return (os.getenv("CARTESIA_API_KEY") or "").strip()


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PredictRequest(BaseModel):
    partial_transcript: str
    session_transcript: str = ""
    history: list[HistoryTurn] = Field(default_factory=list)


class PredictResponse(BaseModel):
    auto_complete: Optional[str] = None
    confidence: float
    options: list[str] = Field(min_length=4, max_length=4)


def _build_llm_client() -> tuple[AsyncOpenAI | None, str | None]:
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not openai_key:
        return None, None
    model = (os.getenv("LLM_MODEL") or LLM_MODEL).strip() or LLM_MODEL
    return AsyncOpenAI(api_key=openai_key), model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    api_key = _api_key()
    if not api_key:
        logger.warning("CARTESIA_API_KEY is not set; WebSocket streams will fail until it is.")
        app.state.cartesia = None
    else:
        app.state.cartesia = AsyncCartesia(api_key=api_key)
    llm, model = _build_llm_client()
    app.state.llm = llm
    app.state.llm_model = model
    if llm is None:
        logger.warning("No OPENAI_API_KEY; /api/predict will return 503.")
    try:
        yield
    finally:
        client: AsyncCartesia | None = getattr(app.state, "cartesia", None)
        if client is not None:
            await client.close()
        llm_client: AsyncOpenAI | None = getattr(app.state, "llm", None)
        if llm_client is not None:
            await llm_client.close()


app = FastAPI(title="stutter-health", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    configured = getattr(app.state, "cartesia", None) is not None
    return {
        "ok": True,
        "encoding": PCM_ENCODING,
        "sample_rate": PCM_SAMPLE_RATE,
        "stt_sample_rate": STT_SAMPLE_RATE,
        "cartesia_configured": configured,
        "cartesia_error": None if configured else "CARTESIA_API_KEY is empty in .env",
        "llm_configured": getattr(app.state, "llm", None) is not None,
        "llm_model": getattr(app.state, "llm_model", None),
    }


def _normalize_prediction(payload: dict[str, Any]) -> PredictResponse:
    options = payload.get("options") or []
    if not isinstance(options, list):
        options = [str(options)]
    words = []
    for item in options:
        word = str(item).strip().split()
        if word:
            words.append(word[0])
    while len(words) < 4:
        words.append("")
    words = words[:4]
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    auto_complete = payload.get("auto_complete")
    if auto_complete is not None:
        auto_complete = str(auto_complete).strip() or None
    if confidence < AUTOCOMPLETE_MIN_CONFIDENCE:
        auto_complete = None
    return PredictResponse(auto_complete=auto_complete, confidence=confidence, options=words)


def _history_messages(history: Sequence[HistoryTurn] | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if not history:
        return messages
    for turn in list(history)[-HISTORY_MAX_TURNS:]:
        content = turn.content.strip()[:HISTORY_MAX_CHARS]
        if not content:
            continue
        messages.append({"role": turn.role, "content": content})
    return messages


def _log_session(
    session: SessionState,
    label: str,
    *,
    manual_raw: str = "",
    ink_piece: str = "",
    ink_final: bool | None = None,
    stutter_predict_sent: bool = False,
) -> None:
    if not SESSION_LOG:
        return
    logger.info(
        "%s\n%s",
        label,
        format_session_state(
            session,
            manual_raw=manual_raw,
            ink_piece=ink_piece,
            ink_final=ink_final,
            stutter_predict_sent=stutter_predict_sent,
        ),
    )


def _predict_user_message(partial_transcript: str, session_transcript: str) -> str:
    scene = (session_transcript or "").strip() or (partial_transcript or "").strip()
    fragment = (partial_transcript or "").strip() or scene
    return (
        f"Unstuttered session sentences (authoritative context):\n{scene}\n\n"
        f"Current unfinished sentence:\n{fragment}\n\n"
        "Predict the next word of the current sentence using every session sentence."
    )


async def complete_partial(
    partial_transcript: str,
    history: Sequence[HistoryTurn] | None = None,
    session_transcript: str = "",
) -> PredictResponse:
    llm: AsyncOpenAI | None = getattr(app.state, "llm", None)
    model: str | None = getattr(app.state, "llm_model", None)
    if llm is None or not model:
        raise HTTPException(status_code=503, detail="LLM is not configured")
    scene = (session_transcript or "").strip()
    if not scene and history:
        scene = " ".join(turn.content.strip() for turn in history if turn.content.strip())
    messages: list[dict[str, str]] = [
        {"role": "system", "content": PREDICT_SYSTEM_PROMPT},
        {"role": "user", "content": _predict_user_message(partial_transcript, scene)},
    ]
    try:
        completion = await llm.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail=_openai_error_detail(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=_openai_error_detail(exc)) from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail=_openai_error_detail(exc)) from exc
    raw = completion.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="LLM returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail="LLM returned invalid JSON")
    return _normalize_prediction(parsed)


def _openai_error_detail(exc: APIError) -> str:
    message = getattr(exc, "message", None) or str(exc)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") or body
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return str(message)


@app.post("/api/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> PredictResponse:
    if SESSION_LOG:
        logger.info(
            "predict request | partial=%r | session=%r",
            (body.partial_transcript or "")[:200],
            (body.session_transcript or "")[:400],
        )
    result = await complete_partial(
        body.partial_transcript,
        history=body.history,
        session_transcript=body.session_transcript,
    )
    if SESSION_LOG:
        logger.info("predict response | options=%s | confidence=%s", result.options, result.confidence)
    return result


@app.post("/api/semantic-gap", response_model=GapResponse)
async def semantic_gap(body: GapRequest) -> GapResponse:
    llm: AsyncOpenAI | None = getattr(app.state, "llm", None)
    model: str | None = getattr(app.state, "llm_model", None)
    if llm is None or not model:
        return fallback_gap(body)
    try:
        result = await analyze_caption_gap(llm, model, body)
        if SESSION_LOG:
            logger.info(
                "semantic-gap | gap=%s caption=%r reason=%r",
                result.gap,
                (result.caption or "")[:200],
                result.reason,
            )
        return result
    except (json.JSONDecodeError, APIError, RateLimitError, AuthenticationError, HTTPException):
        logger.exception("semantic gap analysis failed; keeping caption")
        return fallback_gap(body)


@app.post("/api/voice-clone")
async def voice_clone_endpoint(clip: UploadFile = File(...)) -> dict[str, str]:
    data = await clip.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty_clip")
    voice = await clone_voice(
        _cartesia_client(),
        data=data,
        filename=clip.filename or "clip.wav",
        content_type=clip.content_type or "audio/wav",
    )
    return voice_payload(voice)


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
            sample_rate=STT_SAMPLE_RATE,
            turn_start_threshold=0.5,
            turn_eager_end_threshold=0.3,
            turn_end_threshold=0.05,
            turn_end_timeout_ms=11_200,
        ) as cartesia_ws:

            stutter_predict_sent = False
            stutter_hold_token: str | None = None
            predict_tasks: set[asyncio.Task[None]] = set()
            session_history: list[HistoryTurn] = []
            session = SessionState()
            prior_turns: list[str] = []
            latest_utterance_raw = {"value": ""}
            pause_predict_gen = 0
            pause_predict_task: asyncio.Task[None] | None = None
            logger.info("STT session started (auto ink-2, session_log=%s)", SESSION_LOG)
            _log_session(session, "STT session opened")

            def utterance_raw(turn_text: str) -> str:
                parts = [t.strip() for t in (*prior_turns, turn_text) if t and t.strip()]
                return " ".join(parts)

            async def microphone_to_cartesia() -> None:
                nonlocal stutter_predict_sent, stutter_hold_token, prior_turns
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
                        if control.get("type") == "commit":
                            word = str(control.get("text") or "").strip()
                            if word:
                                session.commit(
                                    word,
                                    sentence_complete=bool(control.get("sentence_complete")),
                                )
                                prior_turns.clear()
                                stutter_predict_sent = False
                                stutter_hold_token = None
                                session_history.clear()
                                session_history.append(
                                    HistoryTurn(role="user", content=session.context)
                                )
                                _log_session(
                                    session,
                                    f"commit from UI | word={word!r} sentence_complete={control.get('sentence_complete')}",
                                    manual_raw=session.sentence,
                                    stutter_predict_sent=stutter_predict_sent,
                                )
                            continue
                        if SESSION_LOG and control.get("type") not in {None, "close"}:
                            logger.info("STT control from browser: %s", control)
                        await cartesia_ws.send(control)
                        if control.get("type") == "close":
                            break
                finally:
                    try:
                        await cartesia_ws.send({"type": "close"})
                    except Exception:
                        logger.debug("Ink-2 close already sent", exc_info=True)

            async def emit_pause_predict(gen: int) -> None:
                raw_full = latest_utterance_raw["value"].strip()
                if not raw_full or gen != pause_predict_gen:
                    return
                raw = current_sentence_stutter_raw(
                    session.completed, raw_full, session.current
                )
                if not raw:
                    return
                context_key = collapse_repetitions(raw)
                current_key = collapse_repetitions(session.current or "")
                hit_now = detect_repetition(raw)
                message: dict[str, Any] = {
                    "type": "pause_predict",
                    "context_key": context_key,
                    "current_key": current_key,
                    "raw_partial": raw,
                    "predict_gen": gen,
                    "repetition_hold": _hold_stutter_repeat_predict(
                        sent=stutter_predict_sent,
                        hold_token=stutter_hold_token,
                        stutter_raw=raw,
                    ),
                    "stutter_token": stutter_hold_token,
                    "active_stutter_token": hit_now.token if hit_now else None,
                }
                try:
                    prediction = await complete_partial(
                        raw,
                        session_transcript=session.context,
                    )
                    if gen != pause_predict_gen:
                        return
                    message["predict"] = prediction.model_dump()
                except HTTPException as exc:
                    message["predict"] = None
                    message["predict_error"] = exc.detail
                except Exception:
                    logger.exception("pause predict failed")
                    message["predict"] = None
                    message["predict_error"] = "predict_failed"
                if websocket.client_state != WebSocketState.CONNECTED:
                    return
                if gen != pause_predict_gen:
                    return
                await websocket.send_json(message)
                if SESSION_LOG:
                    predict = message.get("predict")
                    logger.info(
                        "pause → predict | raw=%r | options=%s",
                        raw[:120],
                        (predict or {}).get("options") if isinstance(predict, dict) else None,
                    )

            def schedule_pause_predict() -> None:
                nonlocal pause_predict_gen, pause_predict_task, latest_utterance_raw
                pause_predict_gen += 1
                gen = pause_predict_gen

                async def run() -> None:
                    try:
                        await asyncio.sleep(PAUSE_PREDICT_SEC)
                    except asyncio.CancelledError:
                        return
                    await emit_pause_predict(gen)

                if pause_predict_task and not pause_predict_task.done():
                    pause_predict_task.cancel()
                pause_predict_task = asyncio.create_task(run())
                predict_tasks.add(pause_predict_task)
                pause_predict_task.add_done_callback(predict_tasks.discard)

            async def cartesia_to_browser() -> None:
                nonlocal stutter_predict_sent, stutter_hold_token, prior_turns
                nonlocal pause_predict_gen, pause_predict_task
                async for event in cartesia_ws:
                    payload = _event_payload(event)
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json(payload)
                    event_type = payload.get("type")
                    transcript = payload.get("transcript")
                    if event_type == "turn.start":
                        session.on_turn_start()
                        if SESSION_LOG:
                            logger.info("STT turn.start | prior_turns=%d", len(prior_turns))
                    if event_type in {"turn.update", "turn.eager_end", "turn.end"} and isinstance(
                        transcript, str
                    ) and transcript:
                        raw = utterance_raw(transcript)
                        session.ingest_audio(collapse_repetitions(raw))
                        label = event_type.replace("turn.", "turn ")
                        _log_session(
                            session,
                            label,
                            manual_raw=raw,
                            ink_piece=transcript,
                            ink_final=None,
                            stutter_predict_sent=stutter_predict_sent,
                        )
                        latest_utterance_raw["value"] = raw
                        stutter_raw = current_sentence_stutter_raw(
                            session.completed, raw, session.current
                        )
                        hit = detect_repetition(stutter_raw)
                        repetition_active = (
                            hit is not None and first_stutter(stutter_raw) is not None
                        )
                        tail_repeat = (
                            hit is not None
                            and trailing_repetition_count(stutter_raw, hit.token)
                            >= STUTTER_PREDICT_MIN_REPEAT
                        )
                        repetition_predict = repetition_active and tail_repeat
                        if (
                            repetition_active
                            and hit is not None
                            and websocket.client_state == WebSocketState.CONNECTED
                        ):
                            repetition_msg = {
                                "type": "repetition",
                                **hit.to_dict(),
                                "session_transcript": session.context,
                                "caption": session.caption,
                                "raw_partial": stutter_raw,
                                "collapsed": collapse_repetitions(stutter_raw),
                                "predict_ready": repetition_predict,
                            }
                            asyncio.create_task(websocket.send_json(repetition_msg))
                            new_stutter_episode = (
                                repetition_predict
                                and hit is not None
                                and (
                                    not stutter_predict_sent
                                    or not _same_stutter_token(
                                        hit.token, stutter_hold_token
                                    )
                                )
                            )
                            if new_stutter_episode:
                                stutter_predict_sent = True
                                stutter_hold_token = hit.token
                                if (
                                    pause_predict_task
                                    and not pause_predict_task.done()
                                ):
                                    pause_predict_task.cancel()
                                pause_predict_gen += 1
                                gen = pause_predict_gen

                                async def run_immediate() -> None:
                                    await emit_pause_predict(gen)

                                pause_predict_task = asyncio.create_task(run_immediate())
                                predict_tasks.add(pause_predict_task)
                                pause_predict_task.add_done_callback(
                                    predict_tasks.discard
                                )
                        if stutter_predict_sent and stutter_hold_token:
                            if (
                                trailing_repetition_count(
                                    stutter_raw, stutter_hold_token
                                )
                                < MIN_REPEAT
                            ):
                                stutter_predict_sent = False
                                stutter_hold_token = None
                        hold_pause_predict = _hold_stutter_repeat_predict(
                            sent=stutter_predict_sent,
                            hold_token=stutter_hold_token,
                            stutter_raw=stutter_raw,
                        )
                        if not hold_pause_predict:
                            schedule_pause_predict()
                    if event_type == "turn.end" and isinstance(transcript, str) and transcript:
                        prior_turns.append(transcript.strip())
                        session.on_turn_end()
                    if session.context:
                        session_history.clear()
                        session_history.append(HistoryTurn(role="user", content=session.context))

            send_task = asyncio.create_task(microphone_to_cartesia())
            recv_task = asyncio.create_task(cartesia_to_browser())
            try:
                await send_task
                await asyncio.wait_for(asyncio.shield(recv_task), timeout=15)
            except asyncio.TimeoutError:
                logger.info("Ink-2 did not close after audio ended; closing STT socket")
            finally:
                extra = [task for task in predict_tasks if not task.done()]
                for task in (send_task, recv_task, *extra):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(send_task, recv_task, *extra, return_exceptions=True)
    except WebSocketDisconnect:
        logger.info("STT client disconnected")
        if SESSION_LOG:
            logger.info("STT session ended (client disconnect)")
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
            contexts: dict[str, Any] = {}
            drains: dict[str, asyncio.Task[None]] = {}

            async def drain_context(context_id: str, ctx: Any) -> None:
                try:
                    async for response in ctx.receive():
                        if response.type == "chunk" and response.audio:
                            await websocket.send_bytes(response.audio)
                        elif response.type == "done":
                            await websocket.send_json({"type": "done", "context_id": context_id})
                        elif response.type == "error":
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "context_id": context_id,
                                    "message": getattr(response, "message", None)
                                    or getattr(response, "title", "tts_error"),
                                }
                            )
                finally:
                    contexts.pop(context_id, None)
                    drains.pop(context_id, None)

            def _open_context(request: dict[str, Any], context_id: str) -> Any:
                ctx = tts_ws.context(
                    context_id,
                    model_id=request.get("model_id") or TTS_MODEL,
                    voice=request.get("voice") or DEFAULT_VOICE_ID,
                    language=request.get("language") or "en",
                    output_format={
                        "container": "raw",
                        "encoding": PCM_ENCODING,
                        "sample_rate": PCM_SAMPLE_RATE,
                    },
                )
                contexts[context_id] = ctx
                drains[context_id] = asyncio.create_task(drain_context(context_id, ctx))
                return ctx

            async def synthesize(request: dict[str, Any]) -> None:
                transcript = request.get("text")
                if transcript is None:
                    transcript = request.get("transcript")
                if not isinstance(transcript, str) or transcript == "":
                    await websocket.send_json({"type": "error", "message": "missing_text"})
                    return
                continue_context = bool(request.get("continue_context"))
                supplied_id = request.get("context_id")
                if continue_context and supplied_id:
                    context_id = str(supplied_id)
                    ctx = contexts.get(context_id)
                    if ctx is None:
                        ctx = _open_context(request, context_id)
                else:
                    context_id = str(supplied_id or uuid.uuid4())
                    ctx = _open_context(request, context_id)
                await ctx.push(transcript, continue_=True, flush=True)
                if continue_context:
                    await websocket.send_json({"type": "continued", "context_id": context_id})
                    return
                await ctx.no_more_inputs()
                drain = drains.get(context_id)
                if drain is not None:
                    await drain

            try:
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
            finally:
                pending = [task for task in drains.values() if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
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

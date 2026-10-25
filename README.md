# StutterHealth

Live dictation assistant for people who stutter. The browser captures speech,
**Cartesia Ink-2** streams captions, and **Sonic TTS** speaks back word choices.
An **OpenAI** model suggests next words after pauses or detected repetitions.
Session state tracks **unstuttered sentences** (collapsed STT + your picks), not
raw Ink audio turns.

## Prerequisites

- Python 3.9+
- [Cartesia API key](https://play.cartesia.ai/keys) (`CARTESIA_API_KEY`)
- [OpenAI API key](https://platform.openai.com/api-keys) for predictions and semantic gap (`OPENAI_API_KEY`)
- Microphone and speakers (Chrome or another modern browser for the demo UI)

## Install

```bash
git clone <your-repo-url>
cd stutter-health

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

Edit **`.env`** locally (never commit it). Required:

| Variable | Purpose |
|----------|---------|
| `CARTESIA_API_KEY` | Ink-2 STT + Sonic TTS |
| `OPENAI_API_KEY` | Pause/repetition word predict + semantic gap |
| `CARTESIA_VOICE_ID` | TTS voice (default: Arlo, masculine en-AU) |
| `LLM_MODEL` | Default `gpt-4o-mini` |

Optional: `SESSION_LOG=1`, `SESSION_LOG_TRUNCATE=1`, `SESSION_LOG_FILE=logs/stutter-health.log`.

## Run the demo

```bash
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

1. Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
2. Hard refresh if you changed the UI (check the small **ui build** tag in the header).
3. Click **Start listening**, allow the microphone.
4. Speak (e.g. *“I would like to order a piece of”*). After a pause or a word repetition (3×), **Suggested Words** appear.
5. Tap a chip to hear it in TTS and add it to the session. Use **Finish** when a full auto-completion is offered.

Health check:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

Expect `ok: true` and `cartesia_configured: true` when `CARTESIA_API_KEY` is set.

## Automated tests and integration loop

Unit tests (no network):

```bash
.venv/bin/python -m unittest discover -s tests -q
```

Full loop (starts server on **8010**, optional clip synthesis, live STT + LLM):

```bash
.venv/bin/python scripts/integration_loop.py --start-server --synthesize
```

Against a server you already run on 8000:

```bash
.venv/bin/python scripts/integration_loop.py --base http://127.0.0.1:8000
```

Session debug logs (when `SESSION_LOG=1`) go to `logs/stutter-health.log`.

## Manual WebSocket clients

```bash
# Replay a test clip through STT
.venv/bin/python scripts/ws_client.py stt data/test_clips/order_uh_piece.pcm

# TTS smoke test
.venv/bin/python scripts/ws_client.py tts "Hello from StutterHealth"
```

## Project layout

| Path | Role |
|------|------|
| `main.py` | FastAPI app: `/ws/stt`, `/ws/tts`, predict, semantic gap |
| `static/index.html` | StutterHealth UI |
| `stutter_health/session.py` | Unstuttered sentence session model |
| `stutter_health/repetition.py` | Word/syllable repetition detection |
| `scripts/integration_loop.py` | CI-style regression loop |
| `data/test_clips/` | PCM fixtures + `manifest.json` |

## Audio format

- Browser mic → server resamples to **16 kHz** mono PCM s16le for Ink-2.
- TTS returns **24 kHz** mono PCM s16le for playback.

## Security

- **Do not commit `.env`** or API keys. Only `.env.example` belongs in git.
- Keys stay on the server; the browser uses WebSockets to this app, not Cartesia directly.

## License

Add your license here if applicable.

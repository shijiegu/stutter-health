# stutter-health

FastAPI baseline for an ultra-low-latency accessibility client. The backend
proxies bidirectional WebSockets to Cartesia **Ink-2** (STT) and **Sonic** (TTS).
Client and server audio is raw PCM 16-bit LE at 24 kHz.

## Setup

```bash
cd ~/Documents/stutter-health
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your Cartesia key in `.env` as `CARTESIA_API_KEY`.

## Run

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). `GET /health` should
report `ok: true` and `cartesia_configured: true` once the key is set.

## Verify the streams

### STT (`/ws/stt`)

1. Open a WebSocket to `ws://127.0.0.1:8000/ws/stt`.
2. Send binary frames of mono PCM s16le @ 24 kHz (microphone or a converted WAV).
3. You should receive JSON events (`connected`, `turn.start`, `turn.update`,
   `turn.end`). Closing the socket should not crash the server.

### TTS (`/ws/tts`)

1. Open a WebSocket to `ws://127.0.0.1:8000/ws/tts`.
2. Send `{"text":"Hello from stutter-health"}`.
3. You should receive binary PCM chunks, then a JSON `{"type":"done"}`.
4. Play the collected bytes with:

```bash
ffplay -f s16le -ar 24000 -ac 1 captured.pcm
```

Disconnecting mid-stream should be caught and logged without taking down uvicorn.

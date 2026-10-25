/**
 * Instant Voice Clone client helpers.
 *
 * Do not call this from the ScriptProcessor / AudioWorklet dictation
 * callback. Buffer or record a clip in a separate flow, then call
 * cloneVoiceFromPcm / cloneVoiceFromWav.
 */
export const CLONE_SAMPLE_RATE = 16000;
export const CLONE_SECONDS = 10;
export const CLONE_BYTES = CLONE_SAMPLE_RATE * 2 * CLONE_SECONDS;

export function encodeWav(pcmBytes, sampleRate) {
  const pcm = new Uint8Array(pcmBytes);
  const buffer = new ArrayBuffer(44 + pcm.length);
  const view = new DataView(buffer);
  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + pcm.length, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, pcm.length, true);
  new Uint8Array(buffer, 44).set(pcm);
  return buffer;
}

export async function cloneVoiceFromWav(wavBuffer, filename = "clip.wav") {
  const body = new FormData();
  body.append("clip", new Blob([wavBuffer], { type: "audio/wav" }), filename);
  const response = await fetch("/api/voice-clone", { method: "POST", body });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function cloneVoiceFromPcm(pcmBytes, sampleRate = CLONE_SAMPLE_RATE) {
  return cloneVoiceFromWav(encodeWav(pcmBytes, sampleRate));
}

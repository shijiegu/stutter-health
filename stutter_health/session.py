"""Listen session: unstuttered sentences, not Cartesia audio turns.

A session is one Start-listening stretch. It is the list of intended
(unstuttered) sentences: collapsed repetitions plus words the user picked.
That list is the LLM context. The dictation box is only the current sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stutter_health.repetition import collapse_repetitions, token_core, tokenize


def _tokens(text: str) -> list[str]:
    return [part for part in (text or "").split() if part]


def append_transcript(
    current: str,
    audio_turn: str,
    incoming: str,
    *,
    new_audio: bool = False,
) -> tuple[str, str, str]:
    """Append incoming speech. Returns (text, audio_turn, delta). Never shrinks text."""
    incoming = (incoming or "").strip()
    current = (current or "").strip()
    audio_turn = "" if new_audio else (audio_turn or "").strip()
    if not incoming:
        return current, audio_turn, ""
    if not current:
        return incoming, incoming, incoming
    if current.endswith(incoming) or incoming == audio_turn:
        return current, incoming if incoming.startswith(audio_turn) or not audio_turn else audio_turn, ""
    words = _tokens(incoming)
    if len(words) >= 4 and (current.startswith(incoming) or incoming in current):
        return current, audio_turn or incoming, ""
    if incoming.startswith(current):
        delta = incoming[len(current) :].strip()
        return (f"{current} {delta}".strip() if delta else current), incoming, delta
    if audio_turn and incoming.startswith(audio_turn):
        delta = incoming[len(audio_turn) :].strip()
        if not delta:
            return current, incoming, ""
        return f"{current} {delta}".strip(), incoming, delta
    return f"{current} {incoming}".strip(), incoming, incoming


def merge_session_transcript(
    session: str,
    ink_turn: str,
    incoming: str,
    *,
    new_turn: bool = False,
) -> tuple[str, str]:
    text, turn, _delta = append_transcript(
        session, ink_turn, incoming, new_audio=new_turn
    )
    return text, turn


def current_sentence_stutter_raw(
    completed: list[str], raw: str, current: str
) -> str:
    """Ink raw for the in-progress sentence only (skip completed-sentence echo in the mic)."""
    raw = (raw or "").strip()
    if not raw or not completed:
        return raw
    cur_words = collapse_repetitions(current or "").split()
    if not cur_words:
        return strip_completed_prefix(completed, raw)
    tokens = tokenize(raw)
    if not tokens:
        return strip_completed_prefix(completed, raw)
    anchor_len = min(3, len(cur_words))
    anchor = [word.lower() for word in cur_words[:anchor_len]]
    start = 0
    for index in range(len(tokens)):
        if index + anchor_len > len(tokens):
            continue
        if all(
            token_core(tokens[index + offset]).lower() == anchor[offset]
            for offset in range(anchor_len)
        ):
            start = index
    if start:
        return " ".join(tokens[start:])
    return strip_completed_prefix(completed, raw)


def strip_completed_prefix(completed: list[str], text: str) -> str:
    """Remove echoed completed sentences from the start of STT (e.g. TTS bleed)."""
    incoming = collapse_repetitions(text or "").strip()
    if not incoming or not completed:
        return incoming
    inc_words = incoming.split()
    changed = True
    while changed and inc_words:
        changed = False
        for sentence in reversed(completed):
            sent = collapse_repetitions(sentence or "").strip()
            if not sent:
                continue
            pre_words = sent.split()
            matched = 0
            while (
                matched < len(pre_words)
                and matched < len(inc_words)
                and inc_words[matched].lower() == pre_words[matched].lower()
            ):
                matched += 1
            strip_n = 0
            if matched >= len(pre_words):
                strip_n = matched
            elif len(pre_words) >= 3 and matched >= 3 and matched < len(pre_words):
                # Partial echo: STT repeats the start of a finished sentence then diverges (e.g. … piece of I …).
                strip_n = matched
            if strip_n > 0:
                inc_words = inc_words[strip_n:]
                incoming = " ".join(inc_words).strip()
                changed = True
                break
    return incoming


def format_session(completed: list[str], current: str) -> str:
    """Unstuttered sentences for the predictor, in order."""
    lines: list[str] = []
    for index, sentence in enumerate(completed, 1):
        text = (sentence or "").strip()
        if text:
            lines.append(f"{index}. {text}")
    current = (current or "").strip()
    if current:
        lines.append(f"{len(lines) + 1}. {current}")
    return "\n".join(lines)


@dataclass
class SessionState:
    completed: list[str] = field(default_factory=list)
    current: str = ""
    caption: str = ""
    audio_turn: str = ""
    expect_new_audio_turn: bool = False
    sentence_complete: bool = False

    @property
    def sentence(self) -> str:
        """Flat unstuttered text: completed sentences plus the current one."""
        parts = [part.strip() for part in self.completed if part.strip()]
        if self.current.strip():
            parts.append(self.current.strip())
        return " ".join(parts)

    @property
    def context(self) -> str:
        return format_session(self.completed, self.current)

    def reset(self) -> None:
        self.completed.clear()
        self.current = ""
        self.caption = ""
        self.audio_turn = ""
        self.expect_new_audio_turn = False
        self.sentence_complete = False

    def on_turn_start(self) -> None:
        self.expect_new_audio_turn = True
        self.audio_turn = ""

    def on_turn_end(self) -> None:
        """Audio buffer closed. Does not complete a sentence."""
        self.expect_new_audio_turn = True

    def set_current_collapsed(self, collapsed: str) -> None:
        """Replace the in-progress unstuttered sentence (manual STT stream)."""
        collapsed = (collapsed or "").strip()
        if not collapsed:
            return
        self.current = collapsed
        self.caption = collapsed

    def ingest_audio(self, incoming: str) -> str:
        incoming = (incoming or "").strip()
        if not incoming:
            return ""
        if self.completed:
            incoming = strip_completed_prefix(self.completed, incoming)
            if not incoming:
                return ""
        new_audio = self.expect_new_audio_turn or self.sentence_complete
        if self.sentence_complete:
            self.finish_current()
        self.current, self.audio_turn, delta = append_transcript(
            self.current,
            self.audio_turn,
            incoming,
            new_audio=new_audio,
        )
        if delta:
            self.caption = f"{self.caption} {delta}".strip() if self.caption else delta
        self.expect_new_audio_turn = False
        self.sentence_complete = False
        return delta

    def commit(self, text: str, *, sentence_complete: bool = False) -> str:
        incoming = (text or "").strip()
        if not incoming:
            return self.sentence
        if self.current and incoming.startswith(self.current):
            extra = incoming[len(self.current) :].strip()
            self.current = incoming
            if extra:
                self.caption = f"{self.caption} {extra}".strip() if self.caption else extra
        else:
            self.current, self.audio_turn, delta = append_transcript(
                self.current, self.audio_turn, incoming, new_audio=True
            )
            if delta:
                self.caption = f"{self.caption} {delta}".strip() if self.caption else delta
        self.audio_turn = ""
        self.expect_new_audio_turn = True
        self.sentence_complete = sentence_complete
        if sentence_complete:
            self.finish_current()
        return self.sentence

    def finish_current(self) -> None:
        text = self.current.strip()
        if text:
            self.completed.append(text)
        self.current = ""
        self.caption = ""
        self.audio_turn = ""
        self.expect_new_audio_turn = True
        self.sentence_complete = False

    def apply_gap(self, new_caption: str) -> None:
        """Split off a finished unstuttered sentence; box shows the new one."""
        shown = (new_caption or "").strip()
        previous = self.current.strip()
        if shown and previous.endswith(shown) and previous != shown:
            head = previous[: -len(shown)].strip()
            if head:
                self.completed.append(head)
            self.current = shown
        elif previous and previous != shown:
            self.completed.append(previous)
            self.current = shown
        else:
            self.current = shown or previous
        self.caption = shown
        self.audio_turn = ""
        self.expect_new_audio_turn = True

"""Browser dictation state machine (mirrors static/index.html for automated tests)."""

from __future__ import annotations

from dataclasses import dataclass, field

from stutter_health.repetition import collapse_repetitions
from stutter_health.session import append_transcript, strip_completed_prefix


def collapse_transcript(text: str) -> str:
    return collapse_repetitions(text)


def collapsed_words(text: str) -> list[str]:
    return [part for part in collapse_transcript(text).split() if part]


@dataclass
class DictationState:
    completed_sentences: list[str] = field(default_factory=list)
    caption: str = ""
    current_unstuttered: str = ""
    stt_locked_prefix: str = ""
    stt_turn_text: str = ""
    picked_tail: str = ""
    dictation_lead: str | None = None
    committed_display: str = ""
    ink_turn: str = ""

    def stt_caption_base(self) -> str:
        turn = (self.stt_turn_text or "").strip()
        if not turn and not (self.stt_locked_prefix or "").strip():
            return ""
        if self.stt_locked_prefix:
            return f"{self.stt_locked_prefix} {turn}".strip()
        return turn

    def full_utterance_collapsed(self) -> str:
        parts = [self.stt_locked_prefix, self.stt_turn_text, self.picked_tail]
        joined = " ".join(part.strip() for part in parts if part and part.strip())
        return collapse_transcript(joined)

    def release_dictation_lead_if_caught_up(self) -> None:
        if self.dictation_lead != "session":
            return
        sess = collapse_transcript(self.current_unstuttered or "")
        stt = self.full_utterance_collapsed()
        if not sess:
            return
        if len(collapsed_words(stt)) >= len(collapsed_words(sess)):
            self.dictation_lead = None

    def append_word_to_line(self, line: str, word: str) -> str:
        line = (line or "").strip()
        word = (word or "").strip()
        if not word:
            return line
        if not line:
            return word
        if line.lower().endswith(word.lower()):
            return line
        if word.lower().startswith(line.lower()):
            return word
        return f"{line} {word}".strip()

    def completed_sentences_display_prefix(self) -> str:
        return " ".join(
            s.strip() for s in self.completed_sentences if (s or "").strip()
        ).strip()

    def current_sentence_dictation_line(self) -> str:
        stt_line = " ".join(
            part
            for part in (self.stt_caption_base(), self.picked_tail)
            if part and part.strip()
        ).strip()
        best = stt_line
        for candidate in (self.current_unstuttered, self.committed_display):
            text = (candidate or "").strip()
            if not text:
                continue
            if len(collapsed_words(text)) >= len(collapsed_words(best)):
                best = text
        return best

    def effective_dictation_line(self) -> str:
        current = self.current_sentence_dictation_line()
        prefix = self.completed_sentences_display_prefix()
        if not prefix:
            return current
        if not current:
            return prefix
        return f"{prefix} {current}".strip()

    def paint_session_dictation(self) -> None:
        self.caption = self.effective_dictation_line()

    def sync_caption_display(self) -> None:
        self.caption = self.effective_dictation_line()

    def show_live_caption(self, transcript: str) -> None:
        turn = (transcript or "").strip()
        if not turn:
            return
        self.stt_turn_text = turn
        self.ink_turn = turn
        self.release_dictation_lead_if_caught_up()
        self.sync_caption_display()

    def ingest_audio(self, collapsed: str) -> None:
        incoming = collapse_transcript(collapsed).strip()
        if self.completed_sentences:
            incoming = strip_completed_prefix(self.completed_sentences, incoming)
        if not incoming:
            return
        current = collapse_transcript(self.current_unstuttered or "").strip()
        inc_words = len(collapsed_words(incoming))
        cur_words = len(collapsed_words(current))
        if not current:
            self.current_unstuttered = incoming
        elif inc_words > cur_words:
            self.current_unstuttered = incoming
        elif current and incoming and current not in incoming and incoming not in current:
            self.current_unstuttered = incoming

    def archive_line_for_next_sentence(self, closed_line: str) -> None:
        line = strip_completed_prefix(self.completed_sentences, closed_line or "").strip()
        if not line:
            line = (closed_line or "").strip()
        if line:
            self.completed_sentences.append(line)
        self.current_unstuttered = ""
        self.stt_locked_prefix = ""
        self.stt_turn_text = ""
        self.picked_tail = ""
        self.committed_display = ""
        self.dictation_lead = None
        self.sync_caption_display()

    def commit_chosen_text(self, text: str, *, sentence_complete: bool = False) -> None:
        text = (text or "").strip()
        if not text:
            return
        replace = bool(
            self.current_unstuttered and text.startswith(self.current_unstuttered)
        )
        picked_delta = ""
        if replace:
            picked_delta = text[len(self.current_unstuttered) :].strip()
            self.current_unstuttered = text
        else:
            merged, _turn, delta = append_transcript(
                self.current_unstuttered,
                self.ink_turn,
                text,
                new_audio=True,
            )
            self.current_unstuttered = merged
            picked_delta = (delta or text).strip()
            word = text.strip()
            cur = (self.current_unstuttered or "").strip()
            if (
                word
                and cur
                and not cur.lower().endswith(word.lower())
                and len(word.split()) == 1
            ):
                self.current_unstuttered = f"{cur} {word}".strip()
                picked_delta = word
            if picked_delta:
                self.picked_tail = (
                    f"{self.picked_tail} {picked_delta}".strip()
                    if self.picked_tail
                    else picked_delta
                )
        self.ink_turn = ""
        visible_base = (self.caption or self.stt_caption_base() or self.current_unstuttered or "").strip()
        word = picked_delta or text
        self.committed_display = self.append_word_to_line(visible_base, word)
        self.caption = self.committed_display
        if sentence_complete and self.current_unstuttered:
            self.current_unstuttered = ""
            self.picked_tail = ""
            self.dictation_lead = None
            self.committed_display = ""
        else:
            self.dictation_lead = "session"
        self.caption = self.effective_dictation_line()

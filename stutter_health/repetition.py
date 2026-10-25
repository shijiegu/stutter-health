"""Detect word and syllable repetition in Ink-2 transcripts.

Simulated stutters in this project often do not pause. They repeat a word or
syllable in-text (My My My, Can-Can-Can, to-to-tomorrow). Cartesia turn-end /
pause detection will miss those, so this module watches the transcript string.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

TOKEN_RE = re.compile(r"[A-Za-z']+(?:-[A-Za-z']+)*")
MIN_REPEAT = 2


@dataclass(frozen=True)
class RepetitionHit:
    kind: str  # "word" or "syllable"
    token: str
    count: int
    collapsed: str
    raw: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text or "")


def _parts(token: str) -> list[str]:
    return [part for part in token.split("-") if part]


def token_core(token: str) -> str:
    parts = _parts(token)
    return parts[-1] if parts else token


def _norm(token: str) -> str:
    return token_core(token).lower()


def collapse_repetitions(text: str) -> str:
    """Drop consecutive duplicate words/syllables so the LLM sees the intended stem."""
    collapsed: list[str] = []
    for token in tokenize(text):
        core = token_core(token)
        if collapsed and collapsed[-1].lower() == core.lower():
            continue
        collapsed.append(core)
    return " ".join(collapsed)


def _hyphen_repeat(token: str) -> tuple[str, int] | None:
    parts = _parts(token)
    if len(parts) < MIN_REPEAT:
        return None
    head = parts[0].lower()
    if not head:
        return None
    count = 0
    for part in parts:
        lowered = part.lower()
        if lowered == head or lowered.startswith(head):
            count += 1
        else:
            return None
    if count >= MIN_REPEAT:
        return token_core(token), count
    return None


def detect_repetition(text: str, min_repeat: int = MIN_REPEAT) -> RepetitionHit | None:
    """Return the latest repetition run, if any."""
    tokens = tokenize(text)
    if not tokens:
        return None

    latest: RepetitionHit | None = None
    collapsed = collapse_repetitions(text)

    for token in tokens:
        hyphen = _hyphen_repeat(token)
        if hyphen is not None:
            core, count = hyphen
            latest = RepetitionHit(
                kind="syllable",
                token=core,
                count=count,
                collapsed=collapsed,
                raw=text,
            )

    index = 0
    while index < len(tokens):
        key = _norm(tokens[index])
        run = 1
        cursor = index + 1
        while cursor < len(tokens) and _norm(tokens[cursor]) == key:
            run += 1
            cursor += 1
        if run >= min_repeat:
            latest = RepetitionHit(
                kind="word",
                token=token_core(tokens[index]),
                count=run,
                collapsed=collapsed,
                raw=text,
            )
        index = cursor

    return latest


def first_stutter(text: str, min_repeat: int = MIN_REPEAT) -> dict[str, object] | None:
    """Walk tokens left to right and report the first prefix that flags a stutter."""
    tokens = tokenize(text)
    for end in range(1, len(tokens) + 1):
        prefix = " ".join(tokens[:end])
        hit = detect_repetition(prefix, min_repeat=min_repeat)
        if hit is None:
            continue
        flagged = tokens[end - 1]
        occurrence = sum(1 for token in tokens[:end] if _norm(token) == _norm(flagged))
        return {
            "token_index": end,
            "token": flagged,
            "occurrence": occurrence,
            "prefix": prefix,
            "kind": hit.kind,
            "count": hit.count,
            "collapsed": hit.collapsed,
        }
    return None


def trailing_repetition_count(text: str, token: str) -> int:
    """How many times ``token`` repeats at the end of ``text`` (live stutter)."""
    key = (token or "").strip().lower()
    if not key:
        return 0
    tokens = tokenize(text)
    count = 0
    for part in reversed(tokens):
        if token_core(part).lower() == key:
            count += 1
        else:
            break
    return count

"""Evaluate stutter-detection latency against synthesized clips."""

from __future__ import annotations

import json
from pathlib import Path

from stutter_health.repetition import first_stutter, tokenize
from stutter_health.synthesize import ROOT_DIR

MANIFEST = ROOT_DIR / "data" / "test_clips" / "manifest.json"


def evaluate_clip(row: dict[str, object]) -> dict[str, object]:
    transcript = str(row["transcript"])
    tokens = tokenize(transcript)
    first = first_stutter(transcript)
    duration_s = None
    sample_rate = int(row.get("sample_rate") or 0)
    nbytes = int(row.get("bytes") or 0)
    if sample_rate and nbytes:
        duration_s = nbytes / (2 * sample_rate)
    token_time_s = None
    if first and duration_s and tokens:
        token_time_s = duration_s * (int(first["token_index"]) / len(tokens))
    return {
        "clip_id": row["clip_id"],
        "stutter_type": row["stutter_type"],
        "tokens": tokens[:12],
        "first_stutter": first,
        "clip_duration_s": duration_s,
        "approx_flag_time_s": token_time_s,
    }


def main() -> None:
    rows = json.loads(MANIFEST.read_text())
    print("Detection uses MIN_REPEAT=2 on growing transcript prefixes.\n")
    for row in rows:
        result = evaluate_clip(row)
        first = result["first_stutter"]
        print(f"== {result['clip_id']} ({result['stutter_type']}) ==")
        print(f"  leading tokens: {result['tokens']}")
        if not first:
            print("  first flag: none")
        else:
            print(
                f"  first flag: {first['occurrence']}-th '{first['token']}' "
                f"(token #{first['token_index']}, {first['kind']} x{first['count']})"
            )
            print(f"  prefix: {first['prefix']!r}")
            if result["approx_flag_time_s"] is not None:
                print(
                    f"  approx audio time: {result['approx_flag_time_s']:.2f}s "
                    f"of {result['clip_duration_s']:.2f}s "
                    "(uniform-token estimate, not forced alignment)"
                )
        print()


if __name__ == "__main__":
    main()

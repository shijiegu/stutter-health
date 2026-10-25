import unittest

from stutter_health.repetition import (
    collapse_repetitions,
    detect_repetition,
    first_stutter,
    trailing_repetition_count,
)
from stutter_health.session import current_sentence_stutter_raw


class RepetitionCollapseTest(unittest.TestCase):
    def test_extra_repeated_words_share_collapsed_stem(self):
        stem = collapse_repetitions("I want to order a a")
        self.assertEqual(stem, collapse_repetitions("I want to order a a a a"))

    def test_word_repetition_detects_from_third_token(self):
        raw = "I want to order a a a"
        hit = detect_repetition(raw)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.kind, "word")
        self.assertEqual(hit.token.lower(), "a")
        self.assertGreaterEqual(hit.count, 3)
        self.assertIsNotNone(first_stutter(raw))

    def test_two_as_not_enough_for_stutter_predict_threshold(self):
        hit = detect_repetition("I want to order a a")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.count, 2)
        hit_three = detect_repetition("I want to order a a a")
        self.assertIsNotNone(hit_three)
        self.assertGreaterEqual(hit_three.count, 3)


    def test_sentence_two_repetition_uses_tail_not_echo(self):
        completed = ["I would like to order a uh piece of cake"]
        raw = (
            "I would like to order a uh uh piece of of of of of of "
            "I would also like like to to to to"
        )
        current = "I would also like to"
        tail = current_sentence_stutter_raw(completed, raw, current)
        self.assertIn("like like", tail)
        self.assertNotIn("piece of of", tail)
        hit = detect_repetition(tail)
        self.assertIsNotNone(hit)
        self.assertGreaterEqual(hit.count, 3)

    def test_echo_only_prefix_does_not_count_as_sentence_two_stutter(self):
        completed = ["I would like to order a uh piece of cake"]
        raw = "I would like to order a uh uh piece of of of of of of I would"
        tail = current_sentence_stutter_raw(completed, raw, "I would")
        hit = detect_repetition(tail)
        self.assertIsNone(hit)


    def test_hold_only_while_stutter_token_at_tail(self):
        from main import _hold_stutter_repeat_predict

        stuttering = "I would like to order uh uh uh"
        end = "I would like to order uh uh uh piece of"
        self.assertGreaterEqual(trailing_repetition_count(stuttering, "uh"), 3)
        self.assertEqual(trailing_repetition_count(end, "uh"), 0)
        self.assertTrue(
            _hold_stutter_repeat_predict(
                sent=True, hold_token="uh", stutter_raw=stuttering
            )
        )
        self.assertFalse(
            _hold_stutter_repeat_predict(sent=True, hold_token="uh", stutter_raw=end)
        )

    def test_fluent_piece_of_after_uh_stutter_allows_pause_predict(self):
        from main import _hold_stutter_repeat_predict

        phrase = "I would like to order uh uh uh piece of"
        self.assertEqual(trailing_repetition_count(phrase, "uh"), 0)
        self.assertFalse(
            _hold_stutter_repeat_predict(sent=True, hold_token="uh", stutter_raw=phrase)
        )


if __name__ == "__main__":
    unittest.main()

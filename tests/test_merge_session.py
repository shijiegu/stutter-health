import unittest

from stutter_health.repetition import collapse_repetitions
from stutter_health.session import SessionState, merge_session_transcript, strip_completed_prefix

from main import _predict_user_message


class MergeSessionTest(unittest.TestCase):
    def test_new_cartesia_turn_keeps_starbucks_and_clicked_coffee(self):
        session, turn = merge_session_transcript(
            "", "", "We are at this starbucks and I would like a small"
        )
        session, turn = merge_session_transcript(session, turn, "coffee", new_turn=True)
        session, turn = merge_session_transcript(
            session,
            turn,
            collapse_repetitions("I I I would also like a"),
            new_turn=True,
        )
        self.assertIn("starbucks", session)
        self.assertIn("coffee", session)
        self.assertTrue(session.endswith("I would also like a"))

    def test_same_turn_grows_without_duplicating(self):
        session, turn = merge_session_transcript("", "", "We are")
        session, turn = merge_session_transcript(session, turn, "We are at this starbucks")
        self.assertEqual(session, "We are at this starbucks")
        self.assertEqual(turn, "We are at this starbucks")

    def test_late_turn_end_after_click_does_not_duplicate(self):
        session, turn = merge_session_transcript(
            "", "", "We are at this starbucks and I would like a small"
        )
        session, turn = merge_session_transcript(session, turn, "coffee", new_turn=True)
        session, turn = merge_session_transcript(
            session,
            turn,
            "We are at this starbucks and I would like a small",
            new_turn=True,
        )
        self.assertEqual(
            session, "We are at this starbucks and I would like a small coffee"
        )

    def test_short_new_turn_does_not_replace_session(self):
        session, turn = merge_session_transcript(
            "We are at this starbucks and I would like a small coffee",
            "We are at this starbucks and I would like a small",
            "I",
            new_turn=True,
        )
        self.assertEqual(
            session, "We are at this starbucks and I would like a small coffee I"
        )


class SessionStateTest(unittest.TestCase):
    def test_set_current_collapsed_for_manual_stream(self):
        session = SessionState()
        session.set_current_collapsed("I would like a piece of cake")
        self.assertEqual(session.current, "I would like a piece of cake")
        self.assertEqual(session.caption, "I would like a piece of cake")

    def test_audio_turn_end_does_not_clear_sentence(self):
        session = SessionState()
        session.ingest_audio("We are at this starbucks and I would like a small")
        session.commit("coffee")
        before = session.sentence
        session.on_turn_end()
        session.on_turn_start()
        self.assertEqual(session.sentence, before)
        self.assertEqual(session.audio_turn, "")
        session.ingest_audio("I would also like a")
        self.assertTrue(session.sentence.startswith(before))
        self.assertIn("starbucks", session.sentence)

    def test_finished_sentence_does_not_duplicate_in_second_ingest(self):
        session = SessionState()
        session.ingest_audio("I would like to order uh piece of")
        session.commit("I would like to order uh piece of cake", sentence_complete=True)
        self.assertEqual(len(session.completed), 1)
        self.assertEqual(session.current, "")
        session.on_turn_start()
        echoed = collapse_repetitions(
            "I would like to order uh piece of cake I would also like to order"
        )
        session.ingest_audio(echoed)
        self.assertEqual(
            session.current, collapse_repetitions("I would also like to order")
        )
        self.assertIn("cake", session.sentence)
        self.assertNotIn("piece of cake I would like", session.current)

    def test_strip_completed_prefix(self):
        completed = ["I would like to order uh piece of cake"]
        raw = collapse_repetitions(
            "I would like to order uh piece of cake I would also like"
        )
        stripped = strip_completed_prefix(completed, raw)
        self.assertEqual(stripped, collapse_repetitions("I would also like"))

    def test_strip_partial_echo_of_completed_sentence(self):
        completed = ["I would like to order a uh piece of cake"]
        raw = collapse_repetitions(
            "I would like to order a uh piece of I would also like to order a cup of"
        )
        stripped = strip_completed_prefix(completed, raw)
        self.assertEqual(
            stripped,
            collapse_repetitions("I would also like to order a cup of"),
        )

    def test_caption_appends_then_gap_resets_window_only(self):
        session = SessionState()
        session.ingest_audio("We are at this starbucks")
        session.ingest_audio("We are at this starbucks and I would like a small")
        self.assertEqual(
            session.caption, "We are at this starbucks and I would like a small"
        )
        session.commit("coffee")
        session.on_turn_start()
        session.ingest_audio("I would also like a")
        self.assertIn("coffee", session.caption)
        self.assertIn("I would also like a", session.caption)
        session.apply_gap("I would also like a")
        self.assertEqual(session.caption, "I would also like a")
        self.assertEqual(session.current, "I would also like a")
        self.assertTrue(any("starbucks" in item for item in session.completed))
        self.assertIn("starbucks", session.sentence)
        self.assertIn("coffee", session.sentence)
        self.assertIn("1.", session.context)
        self.assertIn("2.", session.context)


class PredictContextTest(unittest.TestCase):
    def test_session_is_the_context_window(self):
        text = _predict_user_message(
            "I would also like a",
            "We are at this starbucks and I would like a small coffee I would also like a",
        )
        self.assertIn("starbucks", text)
        self.assertIn("Unstuttered session sentences", text)
        self.assertIn("I would also like a", text)


if __name__ == "__main__":
    unittest.main()

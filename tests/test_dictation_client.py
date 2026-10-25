import unittest

from stutter_health.dictation_sim import DictationState, collapse_transcript


class DictationClientTest(unittest.TestCase):
    def test_picked_word_stays_visible_after_stt_update(self):
        state = DictationState()
        state.stt_locked_prefix = "I would like to order uh uh uh"
        state.stt_turn_text = "piece of"
        state.caption = "I would like to order uh uh uh piece of"
        state.current_unstuttered = collapse_transcript(state.caption)
        state.sync_caption_display()
        self.assertIn("piece of", state.caption)
        self.assertNotIn("cake", state.caption)

        state.commit_chosen_text("cake")
        self.assertIn("cake", state.current_unstuttered)
        self.assertIn("cake", state.caption)
        self.assertIn("cake", state.committed_display)

        state.show_live_caption("piece of")
        self.assertIn("cake", state.caption)
        self.assertIn("piece of", state.caption)

        state.ingest_audio(state.full_utterance_collapsed())
        self.assertIn("cake", state.current_unstuttered)

    def test_new_sentence_after_prediction_pick_can_ingest(self):
        state = DictationState()
        state.caption = "I would like to order uh piece of cake"
        state.current_unstuttered = collapse_transcript(state.caption)
        state.commit_chosen_text("cake")
        state.archive_line_for_next_sentence(state.caption)
        state.stt_turn_text = "I would I would also like to order a cup of of"
        state.ingest_audio(state.full_utterance_collapsed())
        self.assertIn("also", state.current_unstuttered)
        self.assertIn("cup", state.current_unstuttered)

    def test_archived_first_sentence_stays_in_dictation_on_second_stt(self):
        state = DictationState()
        first = "I would like to order uh piece of cake"
        state.archive_line_for_next_sentence(first)
        state.show_live_caption("I would")
        self.assertIn("cake", state.caption)
        self.assertIn("I would", state.caption)
        state.show_live_caption("I would also like")
        self.assertIn("cake", state.caption)
        self.assertIn("also", state.caption)

    def test_ingest_strips_echo_of_completed_sentence(self):
        state = DictationState()
        state.archive_line_for_next_sentence("I would like to order uh piece of cake")
        echoed = collapse_transcript(
            "I would like to order uh piece of I would also like to order a cup of"
        )
        state.ingest_audio(echoed)
        self.assertNotIn("piece of I would", state.current_unstuttered)
        self.assertIn("also", state.current_unstuttered)
        self.assertIn("cup", state.current_unstuttered)
        state.sync_caption_display()
        self.assertIn("cake", state.caption)
        self.assertIn("also", state.caption)
        self.assertNotIn("piece of I would also", state.caption)

    def test_second_archive_keeps_first_sentence_in_caption(self):
        state = DictationState()
        state.archive_line_for_next_sentence("I would like to order uh piece of cake")
        state.current_unstuttered = "I would also like to order uh coffee"
        state.archive_line_for_next_sentence(state.current_unstuttered)
        self.assertIn("cake", state.caption)
        self.assertIn("coffee", state.caption)

    def test_log_style_echo_does_not_pollute_two_sentence_display(self):
        state = DictationState()
        state.archive_line_for_next_sentence("I would like to order a uh piece of cake")
        polluted = collapse_transcript(
            "I would like to order a uh piece of I would also like to order a cup of coffee"
        )
        state.current_unstuttered = polluted
        state.archive_line_for_next_sentence(polluted)
        state.ingest_audio(collapse_transcript("I would also like to order a cup of"))
        state.sync_caption_display()
        self.assertIn("cake", state.caption)
        self.assertIn("coffee", state.caption)
        self.assertNotIn("piece of I would also", state.caption)


if __name__ == "__main__":
    unittest.main()

import queue
import unittest

import numpy as np

from live_subtitles.core import Caption, offer_latest, prepare_audio, timestamp, to_srt


class AudioTests(unittest.TestCase):
    def test_stereo_48khz_becomes_mono_16khz(self):
        t = np.arange(48000) / 48000
        tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        stereo = np.column_stack((tone, tone * 0.5))
        actual = prepare_audio(stereo.tobytes(), 2, 48000)
        self.assertEqual(actual.shape, (16000,))
        self.assertEqual(actual.dtype, np.float32)
        reference = 0.75 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
        np.testing.assert_allclose(actual[100:-100], reference[100:-100], atol=0.002)

    def test_44100_supported_and_silence_stays_silent(self):
        actual = prepare_audio(np.zeros((44100, 2), np.float32).tobytes(), 2, 44100)
        self.assertEqual(len(actual), 16000)
        self.assertFalse(actual.any())

    def test_invalid_samples_do_not_reach_model(self):
        raw = np.array([np.nan, np.inf, -np.inf, 2, -2], np.float32)
        actual = prepare_audio(raw.tobytes(), 1, 16000)
        np.testing.assert_array_equal(actual, [0, 0, 0, 1, -1])

    def test_invalid_format_rejected(self):
        with self.assertRaises(ValueError):
            prepare_audio(b'', 0, 16000)

    def test_backlog_drops_oldest_not_newest(self):
        pending = queue.Queue(maxsize=2)
        self.assertFalse(offer_latest(pending, ('old', 1)))
        self.assertFalse(offer_latest(pending, ('middle', 2)))
        self.assertTrue(offer_latest(pending, ('latest', 3)))
        self.assertEqual(pending.get_nowait(), ('middle', 2))
        self.assertEqual(pending.get_nowait(), ('latest', 3))


class SubtitleTests(unittest.TestCase):
    def test_srt_preserves_bilingual_content_and_timing(self):
        captions = [Caption(1.25, 4.5, 'Hello world.', '世界你好。'),
                    Caption(8, 9, 'Thank you.', '謝謝。')]
        self.assertEqual(to_srt(captions),
                         '1\n00:00:01,250 --> 00:00:04,500\n世界你好。\nHello world.\n\n'
                         '2\n00:00:08,000 --> 00:00:09,000\n謝謝。\nThank you.\n')

    def test_timestamp_carries_millisecond_rounding(self):
        self.assertEqual(timestamp(59.9999), '00:01:00,000')
        self.assertEqual(timestamp(3661.001), '01:01:01,001')
        self.assertEqual(to_srt([]), '')


if __name__ == '__main__':
    unittest.main()

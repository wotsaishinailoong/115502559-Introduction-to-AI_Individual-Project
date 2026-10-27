import queue
import threading
import unittest
from unittest.mock import patch

import numpy as np

from live_subtitles.app import App
from live_subtitles.core import Caption


class PipelineTests(unittest.TestCase):
    def make_app(self):
        app = App.__new__(App)
        app.stop = threading.Event()
        app.events = queue.Queue()
        app.models = None
        return app

    def test_audio_to_caption_keeps_capture_offset(self):
        app = self.make_app()
        translated = threading.Event()

        class Models:
            def recognize(self, audio):
                self.audio = audio
                return [(0.5, 1.5, 'Hello.')]

            def translate(self, text):
                self.text = text
                return '你好。'

        app.models = Models()
        audio = np.ones(16000, np.float32)

        def capture(index, pending, stop, emit):
            pending.put((12, audio))
            translated.wait(2)
            stop.set()

        def emit(kind, value):
            app.events.put((kind, value))
            if kind == 'caption':
                translated.set()

        app.emit = emit
        with patch('live_subtitles.app.capture', capture):
            app.run(0)
        self.assertEqual(app.events.get_nowait(), ('caption', Caption(12.5, 13.5, 'Hello.', '你好。')))
        self.assertEqual(app.events.get_nowait(), ('done', None))
        np.testing.assert_array_equal(app.models.audio, audio)
        self.assertEqual(app.models.text, 'Hello.')

    def test_model_failure_reports_error_and_stops(self):
        app = self.make_app()
        with patch('live_subtitles.app.LocalModels', side_effect=RuntimeError('download unavailable')):
            app.run(0)
        kind, error = app.events.get_nowait()
        self.assertEqual(kind, 'error')
        self.assertIn('download unavailable', error)
        self.assertTrue(app.stop.is_set())
        self.assertEqual(app.events.get_nowait(), ('done', None))

    def test_cancel_during_load_does_not_open_audio(self):
        app = self.make_app()
        def load(status):
            app.stop.set()
            return object()
        with patch('live_subtitles.app.LocalModels', load), patch('live_subtitles.app.capture') as capture:
            app.run(0)
        capture.assert_not_called()
        self.assertEqual(app.events.get_nowait(), ('done', None))

import os
from pathlib import Path


class LocalModels:
    def __init__(self, status):
        # Imports and model loading stay on the worker thread, never on the UI.
        from faster_whisper import WhisperModel
        from transformers import MarianMTModel, MarianTokenizer
        from opencc import OpenCC
        import torch

        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) - 1)))
        cache = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'LocalSubtitles' / 'models'
        cache.mkdir(parents=True, exist_ok=True)
        status('載入英文辨識模型（首次需下載，可能需要數分鐘）…')
        self.whisper = WhisperModel('base.en', device='cpu', compute_type='int8',
                                    download_root=str(cache / 'whisper'))
        status('載入中譯模型（首次需下載）…')
        name = 'Helsinki-NLP/opus-mt-en-zh'
        self.tokenizer = MarianTokenizer.from_pretrained(name, cache_dir=str(cache / 'translation'))
        self.translator = MarianMTModel.from_pretrained(name, cache_dir=str(cache / 'translation'))
        self.translator.eval()
        self.converter = OpenCC('s2twp')

    def recognize(self, audio):
        segments, _ = self.whisper.transcribe(
            audio, language='en', beam_size=3, vad_filter=True,
            condition_on_previous_text=False,
        )
        return [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]

    def translate(self, text):
        import torch
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.inference_mode():
            result = self.translator.generate(**inputs, max_new_tokens=256)
        return self.converter.convert(self.tokenizer.decode(result[0], skip_special_tokens=True))

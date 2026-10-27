from dataclasses import dataclass
from math import gcd
from queue import Empty, Full

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class Caption:
    start: float
    end: float
    english: str
    chinese: str


def prepare_audio(raw: bytes, channels: int, rate: int) -> np.ndarray:
    """Convert interleaved float32 WASAPI audio into Whisper's 16 kHz mono."""
    if channels < 1 or rate < 1:
        raise ValueError('Invalid audio format')
    samples = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels)
    mono = np.nan_to_num(samples.mean(axis=1), nan=0, posinf=0, neginf=0)
    if rate != 16000 and mono.size:
        divisor = gcd(rate, 16000)
        mono = resample_poly(mono, 16000 // divisor, rate // divisor)
    return np.clip(mono, -1, 1).astype(np.float32)


def offer_latest(queue, item):
    """Bound latency: discard oldest pending audio when inference falls behind."""
    dropped = False
    try:
        queue.put_nowait(item)
    except Full:
        try:
            queue.get_nowait()
            dropped = True
        except Empty:
            pass
        queue.put_nowait(item)
    return dropped


def timestamp(seconds):
    milliseconds = max(0, round(seconds * 1000))
    seconds, ms = divmod(milliseconds, 1000)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f'{hours:02}:{minute:02}:{sec:02},{ms:03}'


def to_srt(captions):
    return '\n\n'.join(
        f'{i}\n{timestamp(c.start)} --> {timestamp(c.end)}\n{c.chinese}\n{c.english}'
        for i, c in enumerate(captions, 1)
    ) + ('\n' if captions else '')

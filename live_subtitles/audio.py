import sys
import time

from .core import offer_latest, prepare_audio


def devices():
    if sys.platform != 'win32':
        raise RuntimeError('系統音訊擷取需要 Windows 10／11。')
    import pyaudiowpatch as pyaudio
    with pyaudio.PyAudio() as driver:
        choices = list(driver.get_loopback_device_info_generator())
        try:
            default = driver.get_default_wasapi_loopback()['index']
        except (OSError, LookupError):
            default = choices[0]['index'] if choices else None
        return choices, default


def capture(device_index, queue, stop, emit):
    import pyaudiowpatch as pyaudio
    try:
        with pyaudio.PyAudio() as driver:
            info = driver.get_device_info_by_index(device_index)
            rate = int(info['defaultSampleRate'])
            channels = int(info['maxInputChannels'])
            # Four-second chunks keep CPU usage manageable. Capture runs independently
            # of inference, so model processing never blocks the audio reader.
            target = rate * 4
            with driver.open(format=pyaudio.paFloat32, channels=channels, rate=rate,
                             input=True, input_device_index=device_index,
                             frames_per_buffer=1024) as stream:
                emit('status', '正在聆聽系統音訊；請播放英文內容。')
                frames = 0
                parts = []
                started = time.monotonic()
                chunk_start = 0.0
                while not stop.is_set():
                    raw = stream.read(1024, exception_on_overflow=True)
                    if not parts:
                        chunk_start = max(0, time.monotonic() - started - 1024 / rate)
                    parts.append(raw)
                    frames += len(raw) // (4 * channels)
                    if frames >= target:
                        audio = prepare_audio(b''.join(parts), channels, rate)
                        if offer_latest(queue, (chunk_start, audio)):
                            emit('status', '電腦處理速度不足，已略過較舊音訊以降低延遲。')
                        parts, frames = [], 0
    except Exception as exc:
        emit('error', f'音訊擷取失敗：{exc}')
        stop.set()

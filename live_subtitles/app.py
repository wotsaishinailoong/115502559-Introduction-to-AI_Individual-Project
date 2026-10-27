import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .audio import capture, devices
from .core import Caption, to_srt
from .models import LocalModels


class App:
    def __init__(self, root):
        self.root = root
        root.title('英文系統音訊 → 繁體中文字幕｜本機版')
        root.geometry('900x640')
        root.minsize(680, 480)
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.worker = None
        self.models = None
        self.captions = []
        self.closing = False
        self.status = tk.StringVar(value='選擇音源後按「開始」。首次執行需要連網下載模型。')
        frame = ttk.Frame(root, padding=18)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='即時英文聆聽・繁體中文翻譯', font=('Microsoft JhengHei UI', 20, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='擷取喇叭／耳機播放內容，不使用麥克風；音訊於本機處理。').pack(anchor='w', pady=(6, 16))
        controls = ttk.Frame(frame)
        controls.pack(fill='x')
        self.device = ttk.Combobox(controls, state='readonly', width=48)
        self.device.pack(side='left', fill='x', expand=True)
        self.refresh_button = ttk.Button(controls, text='重新偵測', command=self.refresh)
        self.refresh_button.pack(side='left', padx=6)
        self.start_button = ttk.Button(controls, text='開始', command=self.start)
        self.start_button.pack(side='left')
        self.stop_button = ttk.Button(controls, text='停止', command=self.stop_run, state='disabled')
        self.stop_button.pack(side='left', padx=6)
        self.text = ScrolledText(frame, wrap='word', font=('Microsoft JhengHei UI', 14), state='disabled')
        self.text.pack(fill='both', expand=True, pady=16)
        self.text.tag_configure('english', foreground='#64748b', font=('Segoe UI', 11))
        footer = ttk.Frame(frame)
        footer.pack(fill='x')
        ttk.Button(footer, text='匯出雙語 SRT', command=self.export).pack(side='right')
        ttk.Label(footer, textvariable=self.status, wraplength=630).pack(side='left', fill='x')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.refresh()
        root.after(100, self.poll)

    def refresh(self):
        try:
            self.choices, default = devices()
            self.device['values'] = [x['name'] for x in self.choices]
            if not self.choices:
                raise RuntimeError('找不到 WASAPI loopback 音源，請確認喇叭／耳機已連接。')
            self.device.current(next((i for i, x in enumerate(self.choices) if x['index'] == default), 0))
            self.start_button['state'] = 'normal'
        except Exception as exc:
            self.choices = []
            self.status.set(str(exc))
            self.start_button['state'] = 'disabled'

    def emit(self, kind, value):
        self.events.put((kind, value))

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        index = self.device.current()
        if index < 0:
            return
        if self.captions and not messagebox.askyesno('開始新一輪', '開始後會清除目前字幕。需要保留時，請先取消並匯出。'):
            return
        self.captions.clear()
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        self.text.configure(state='disabled')
        self.stop.clear()
        self.start_button['state'] = self.refresh_button['state'] = self.device['state'] = 'disabled'
        self.stop_button['state'] = 'normal'
        self.status.set('準備模型…')
        self.worker = threading.Thread(target=self.run, args=(self.choices[index]['index'],), daemon=True)
        self.worker.start()

    def run(self, device_index):
        recorder = None
        try:
            if self.models is None:
                self.models = LocalModels(lambda text: self.emit('status', text))
            if self.stop.is_set():
                return
            audio_queue = queue.Queue(maxsize=2)
            recorder = threading.Thread(target=capture, args=(device_index, audio_queue, self.stop, self.emit), daemon=True)
            recorder.start()
            while not self.stop.is_set():
                try:
                    offset, audio = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                for start, end, english in self.models.recognize(audio):
                    if self.stop.is_set():
                        break
                    chinese = self.models.translate(english)
                    if not self.stop.is_set():
                        self.emit('caption', Caption(offset + start, offset + end, english, chinese))
        except Exception as exc:
            self.emit('error', f'辨識／翻譯失敗：{exc}\n首次執行請確認網路可下載 Hugging Face 模型；詳見 README。')
        finally:
            self.stop.set()
            if recorder:
                recorder.join(timeout=3)
            self.emit('done', None)

    def stop_run(self):
        self.stop.set()
        self.stop_button['state'] = 'disabled'
        self.status.set('正在停止，等待目前模型作業結束…')

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'caption':
                    self.captions.append(value)
                    self.text.configure(state='normal')
                    self.text.insert('end', value.chinese + '\n')
                    self.text.insert('end', value.english + '\n\n', 'english')
                    self.text.see('end')
                    self.text.configure(state='disabled')
                elif kind == 'error':
                    self.status.set('發生錯誤，已停止。')
                    if not self.closing:
                        messagebox.showerror('無法繼續', value)
                elif kind == 'status':
                    if not self.stop.is_set():
                        self.status.set(value)
                elif kind == 'done':
                    self.start_button['state'] = self.refresh_button['state'] = 'normal'
                    self.device['state'] = 'readonly'
                    self.stop_button['state'] = 'disabled'
                    self.status.set('已停止，可匯出字幕或重新開始。')
        except queue.Empty:
            pass
        if self.closing and (not self.worker or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(100, self.poll)

    def export(self):
        if not self.captions:
            messagebox.showinfo('匯出字幕', '目前還沒有字幕。')
            return
        path = filedialog.asksaveasfilename(defaultextension='.srt', filetypes=[('SubRip subtitles', '*.srt')])
        if path:
            try:
                with open(path, 'w', encoding='utf-8-sig') as output:
                    output.write(to_srt(self.captions))
            except OSError as exc:
                messagebox.showerror('匯出失敗', str(exc))

    def close(self):
        self.closing = True
        self.stop_run()
        self.root.withdraw()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()

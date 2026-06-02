#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive MNIST digit recognition GUI.
Draw digits on canvas, get real-time CNN predictions.
Supports single digit and multi-digit string recognition.

Usage:
    python mnist_gui.py
"""
import os as _os
_os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageTk
import tkinter as tk
from tkinter import ttk
import io
import os
import sys

# ── CNN Model ───────────────────────────────────────────
class MNIST_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv2(self.conv1(x)))


# ── Load Model ──────────────────────────────────────────
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNIST_CNN().to(device)

    # Resolve script directory robustly
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    model_paths = [
        os.path.join(script_dir, 'results', 'mnist_cnn_gpu.pth'),
        os.path.join(script_dir, 'results', 'mnist_cnn.pth'),
        os.path.join(os.getcwd(), 'results', 'mnist_cnn_gpu.pth'),
        os.path.join(os.getcwd(), 'results', 'mnist_cnn.pth'),
    ]
    # Deduplicate
    seen = set()
    unique_paths = []
    for p in model_paths:
        p = os.path.normpath(p)
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    for path in unique_paths:
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location=device, weights_only=True)
                model.load_state_dict(state)
                print(f"Model loaded: {path}")
                model.eval()
                return model, device
            except Exception as e:
                print(f"Warning: failed to load {path}: {e}")
                continue

    # If we get here, no model was found
    searched = '\n  '.join(unique_paths)
    import tkinter.messagebox as mb
    mb.showerror("模型未找到",
                 f"请先训练模型:\n\n  python cnn_train_gpu.py\n\n已搜索:\n  {searched}")
    sys.exit(1)


model = None
device = None


# ── Prediction helpers ──────────────────────────────────
def preprocess(img_pil):
    """Convert PIL image to normalized tensor."""
    img = img_pil.resize((28, 28), Image.LANCZOS).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)


def predict_digit(img_pil):
    tensor = preprocess(img_pil)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return pred, probs


def segment_digits(img_pil, min_width=4):
    """Segment wide image into individual digits by vertical projection."""
    img = img_pil.convert('L')
    img_inv = ImageOps.invert(img)
    arr = np.array(img_inv, dtype=np.float32)
    col_sums = arr.sum(axis=0)
    threshold = max(col_sums.max() * 0.03, 15)
    ink = col_sums > threshold
    segments = []
    in_digit = False
    start = 0
    for i, has_ink in enumerate(ink):
        if has_ink and not in_digit:
            start = i; in_digit = True
        elif not has_ink and in_digit:
            if i - start >= min_width:
                pad = 3
                x1, x2 = max(0, start - pad), min(img.width, i + pad)
                segments.append(img.crop((x1, 0, x2, img.height)))
            in_digit = False
    if in_digit and len(ink) - start >= min_width:
        x1, x2 = max(0, start - 3), min(img.width, len(ink) + 3)
        segments.append(img.crop((x1, 0, x2, img.height)))
    return segments


# ── PIL-backed drawing widget ───────────────────────────
class DrawPad(tk.Frame):
    """A drawing pad backed by a PIL Image."""

    def __init__(self, parent, width, height, brush=12, bg='black', fg='white', **kw):
        super().__init__(parent, bg='#1e1e2e', **kw)
        self.w = width
        self.h = height
        self.brush = brush
        self.bg = bg
        self.fg = fg
        self._on_draw_callback = None

        # Backing PIL image
        self.image = Image.new('L', (width, height), color=0 if bg == 'black' else 255)
        self.draw_ctx = ImageDraw.Draw(self.image)

        # Tk display
        self.photo = ImageTk.PhotoImage(self.image)
        self.label = tk.Label(self, image=self.photo, bg='#1e1e2e', cursor='cross')
        self.label.pack()
        self.label.bind('<B1-Motion>', self._draw)
        self.label.bind('<ButtonPress-1>', self._draw)
        self.label.bind('<ButtonRelease-1>', self._on_release)

    def set_draw_callback(self, cb):
        self._on_draw_callback = cb

    def _draw(self, event):
        x, y = event.x, event.y
        r = self.brush // 2
        self.draw_ctx.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self._update_display()

    def _on_release(self, event):
        if self._on_draw_callback:
            self._on_draw_callback()

    def _update_display(self):
        self.photo = ImageTk.PhotoImage(self.image)
        self.label.configure(image=self.photo)

    def get_image(self):
        return self.image.copy()

    def clear(self):
        self.draw_ctx.rectangle([0, 0, self.w, self.h], fill=0)
        self._update_display()


# ── Main App ────────────────────────────────────────────
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST 手写数字识别 — 交互测试")
        self.root.geometry("920x720")
        self.root.configure(bg='#1e1e2e')
        self.root.minsize(850, 620)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#1e1e2e', borderwidth=0)
        style.configure('TNotebook.Tab', padding=[24, 9], font=('Microsoft YaHei', 11),
                        background='#2d2d44', foreground='#a0aec0')
        style.map('TNotebook.Tab', background=[('selected', '#1e1e2e')],
                  foreground=[('selected', '#ffffff')])

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.tab1 = tk.Frame(self.notebook, bg='#1e1e2e')
        self.tab2 = tk.Frame(self.notebook, bg='#1e1e2e')
        self.notebook.add(self.tab1, text='  单个数字  ')
        self.notebook.add(self.tab2, text='  数字串  ')

        self._build_single()
        self._build_multi()

    # ═══ Single Digit Tab ═══════════════════════════════
    def _build_single(self):
        mf = tk.Frame(self.tab1, bg='#1e1e2e')
        mf.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        left = tk.Frame(mf, bg='#1e1e2e')
        left.pack(side=tk.LEFT, padx=(0, 24))
        tk.Label(left, text='用鼠标在这里写数字', font=('Microsoft YaHei', 13, 'bold'),
                 fg='#a0aec0', bg='#1e1e2e').pack(pady=(0, 8))
        self.pad1 = DrawPad(left, 300, 300, brush=18, bg='black', fg='white')
        self.pad1.pack()
        self.pad1.set_draw_callback(self._predict_single)
        bf = tk.Frame(left, bg='#1e1e2e')
        bf.pack(fill=tk.X, pady=10)
        tk.Button(bf, text='清除', font=('Microsoft YaHei', 12), bg='#3a3a5c', fg='white',
                  relief=tk.FLAT, cursor='hand2', activebackground='#505078',
                  padx=18, pady=5, command=self._clear_single).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text='识别', font=('Microsoft YaHei', 12), bg='#34c759', fg='white',
                  relief=tk.FLAT, cursor='hand2', activebackground='#28a745',
                  padx=18, pady=5, command=self._predict_single).pack(side=tk.LEFT, padx=4)

        # Right panel
        right = tk.Frame(mf, bg='#2d2d44', width=380)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right.pack_propagate(False)
        tk.Label(right, text='识别结果', font=('Microsoft YaHei', 15, 'bold'),
                 fg='#fff', bg='#2d2d44').pack(pady=(24, 4))
        self.single_digit = tk.Label(right, text='?', font=('Arial', 80, 'bold'),
                                     fg='#34c759', bg='#2d2d44')
        self.single_digit.pack(pady=8)
        self.single_conf = tk.Label(right, text='在左侧画布上写一个数字',
                                    font=('Microsoft YaHei', 11), fg='#6e6e73', bg='#2d2d44')
        self.single_conf.pack()
        tk.Label(right, text='Top-10 概率', font=('Microsoft YaHei', 12, 'bold'),
                 fg='#a0aec0', bg='#2d2d44').pack(pady=(24, 8))
        self.bars_frame = tk.Frame(right, bg='#2d2d44')
        self.bars_frame.pack(fill=tk.X, padx=28)
        self.s_bars = []
        for i in range(10):
            row = tk.Frame(self.bars_frame, bg='#2d2d44')
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=str(i), font=('Arial', 9, 'bold'), fg='#a0aec0',
                     bg='#2d2d44', width=2, anchor='e').pack(side=tk.LEFT, padx=(0, 6))
            bar = tk.Frame(row, bg='#34c759', width=0, height=15)
            bar.pack(side=tk.LEFT); bar.pack_propagate(False)
            pct = tk.Label(row, text='', font=('Arial', 8), fg='#a0aec0',
                           bg='#2d2d44', width=7, anchor='w')
            pct.pack(side=tk.LEFT, padx=4)
            self.s_bars.append((bar, pct))

    def _clear_single(self):
        self.pad1.clear()
        self.single_digit.config(text='?')
        self.single_conf.config(text='在左侧画布上写一个数字')
        for bar, pct in self.s_bars:
            bar.config(width=0); pct.config(text='')

    def _predict_single(self, event=None):
        img = self.pad1.get_image()
        pred, probs = predict_digit(img)
        self.single_digit.config(text=str(pred))
        self.single_conf.config(text=f'置信度: {probs[pred]*100:.1f}%')
        max_w = 240
        ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        for digit, prob in enumerate(probs):
            bar, pct = self.s_bars[digit]
            bar.config(bg='#34c759' if digit == pred else '#4a4a6a',
                       width=int(prob * max_w))
            pct.config(text=f'{prob*100:.0f}%' if prob > 0.01 else '')

    # ═══ Multi-Digit Tab ════════════════════════════════
    def _build_multi(self):
        mf = tk.Frame(self.tab2, bg='#1e1e2e')
        mf.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(mf, text='写一串数字 (数字之间请保持一定间距)',
                 font=('Microsoft YaHei', 13, 'bold'), fg='#a0aec0', bg='#1e1e2e').pack(pady=(0, 8))
        self.pad2 = DrawPad(mf, 750, 180, brush=14, bg='black', fg='white')
        self.pad2.pack()
        bf = tk.Frame(mf, bg='#1e1e2e')
        bf.pack(fill=tk.X, pady=10)
        tk.Button(bf, text='清除', font=('Microsoft YaHei', 12), bg='#3a3a5c', fg='white',
                  relief=tk.FLAT, cursor='hand2', activebackground='#505078',
                  padx=18, pady=5, command=self._clear_multi).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text='识别数字串', font=('Microsoft YaHei', 12, 'bold'), bg='#ff9500',
                  fg='white', relief=tk.FLAT, cursor='hand2', activebackground='#e68600',
                  padx=18, pady=5, command=self._predict_multi).pack(side=tk.LEFT, padx=4)

        # Results
        bottom = tk.Frame(mf, bg='#2d2d44')
        bottom.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        tk.Label(bottom, text='识别结果', font=('Microsoft YaHei', 14, 'bold'),
                 fg='#fff', bg='#2d2d44').pack(pady=(14, 4))
        self.multi_result = tk.Label(bottom, text='--', font=('Arial', 48, 'bold'),
                                     fg='#ff9500', bg='#2d2d44')
        self.multi_result.pack(pady=4)
        self.multi_hint = tk.Label(bottom, text='点击"识别数字串"按钮进行识别',
                                   font=('Microsoft YaHei', 10), fg='#6e6e73', bg='#2d2d44')
        self.multi_hint.pack()
        self.seg_frame = tk.Frame(bottom, bg='#2d2d44')
        self.seg_frame.pack(pady=10)

    def _clear_multi(self):
        self.pad2.clear()
        self.multi_result.config(text='--')
        self.multi_hint.config(text='点击"识别数字串"按钮进行识别')
        for w in self.seg_frame.winfo_children():
            w.destroy()

    def _predict_multi(self):
        img = self.pad2.get_image()
        segments = segment_digits(img)
        for w in self.seg_frame.winfo_children():
            w.destroy()
        if not segments:
            self.multi_result.config(text='未检测到')
            self.multi_hint.config(text='请检查数字间距是否足够，笔画是否清晰')
            return
        results = []
        for i, seg in enumerate(segments):
            pred, probs = predict_digit(seg)
            results.append(str(pred))
            sf = tk.Frame(self.seg_frame, bg='#3a3a5c', relief=tk.RAISED, bd=1)
            sf.pack(side=tk.LEFT, padx=6, pady=4)
            seg_p = seg.resize((36, 50), Image.LANCZOS)
            photo = ImageTk.PhotoImage(seg_p)
            il = tk.Label(sf, image=photo, bg='#3a3a5c')
            il.image = photo
            il.pack(pady=(4, 1))
            tk.Label(sf, text=f'→ {pred}', font=('Arial', 14, 'bold'),
                     fg='#ff9500', bg='#3a3a5c').pack(pady=(0, 4))
        result_str = ''.join(results)
        self.multi_result.config(text=result_str)
        self.multi_hint.config(text=f'检测到 {len(results)} 个数字')
        print(f"Multi-digit: {len(results)} segments → {result_str}")


def main():
    global model, device
    try:
        # Create window FIRST so user sees it immediately
        root = tk.Tk()
        root.title("MNIST 手写数字识别")
        root.geometry("400x200")
        root.configure(bg='#1e1e2e')

        # Show loading status
        status_label = tk.Label(root, text="正在加载模型...", font=('Microsoft YaHei', 14),
                                fg='#ffffff', bg='#1e1e2e')
        status_label.pack(expand=True)
        root.update()

        # Now load model (user can see the window)
        print("Loading model...")
        model, device = load_model()
        print(f"Device: {device}")

        # Replace loading label with actual app
        status_label.destroy()
        root.destroy()

        # Recreate full window
        root = tk.Tk()
        app = DigitRecognizerApp(root)
        root.bind('<Return>', lambda _e: app._predict_single())
        root.bind('<Delete>', lambda _e: app._clear_single())
        root.mainloop()
    except Exception:
        import traceback
        err_msg = traceback.format_exc()
        print(err_msg)
        try:
            import tkinter.messagebox as mb
            mb.showerror("程序启动失败", err_msg)
        except Exception:
            pass
        sys.exit(1)


if __name__ == '__main__':
    main()

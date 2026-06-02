#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive MNIST digit recognition GUI.
Draw digits on canvas, get real-time CNN predictions.
Supports single digit and multi-digit string recognition.
Multiple model selection via dropdown.

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
import json
import glob


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


# ── Model Manager ───────────────────────────────────────
class ModelManager:
    def __init__(self):
        self.model = None
        self.device = None
        self.current_model_name = ""
        self.available_models = {}  # display_name -> file_path
        self.model_info = {}        # display_name -> {acc, params, ...}
        self._scan_models()

    def _scan_models(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
        results_dir = os.path.join(script_dir, 'results')
        if not os.path.isdir(results_dir):
            results_dir = os.path.join(os.getcwd(), 'results')

        self.available_models.clear()
        self.model_info.clear()

        if os.path.isdir(results_dir):
            for pth_file in sorted(glob.glob(os.path.join(results_dir, '*.pth'))):
                fname = os.path.basename(pth_file)
                # Derive display name from filename
                if 'gpu_v2' in fname:
                    display = 'GPU v2 (batch=64, epoch=25)'
                elif 'gpu' in fname:
                    display = 'GPU v1 (batch=512, epoch=20)'
                elif 'cnn' in fname:
                    display = 'CPU (batch=128, epoch=15)'
                else:
                    display = fname.replace('.pth', '')
                self.available_models[display] = pth_file
                # Try to load accuracy from corresponding log
                log_name = fname.replace('.pth', '')
                for log_ext in [f'training_log_{log_name.replace("mnist_cnn_", "")}.json',
                                f'training_log_{log_name.replace("mnist_", "")}.json',
                                'training_log.json', 'training_log_gpu.json',
                                'training_log_gpu_v2.json']:
                    log_path = os.path.join(results_dir, log_ext)
                    if os.path.exists(log_path):
                        try:
                            with open(log_path) as f:
                                d = json.load(f)
                            self.model_info[display] = {
                                'acc': d.get('best_acc', '?'),
                                'params': d.get('config', {}).get('params', '?'),
                            }
                        except Exception:
                            pass
                        break

    def load_model(self, display_name):
        if display_name not in self.available_models:
            return False
        path = self.available_models[display_name]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = MNIST_CNN().to(self.device)
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
            m.load_state_dict(state)
            m.eval()
            self.model = m
            self.current_model_name = display_name
            print(f"Model loaded: {display_name}  ({path})")
            return True
        except Exception as e:
            print(f"Failed to load {display_name}: {e}")
            import tkinter.messagebox as mb
            mb.showerror("模型加载失败", f"无法加载模型:\n{display_name}\n\n{str(e)}")
            return False

    def get_model_list(self):
        return list(self.available_models.keys())

    def get_info(self, display_name):
        return self.model_info.get(display_name, {})


# Global instance
model_mgr = ModelManager()


# ── Prediction helpers ──────────────────────────────────
def preprocess(img_pil):
    img = img_pil.resize((28, 28), Image.LANCZOS).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(model_mgr.device)


def predict_digit(img_pil):
    tensor = preprocess(img_pil)
    with torch.no_grad():
        output = model_mgr.model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return pred, probs


def segment_digits(img_pil, min_width=4):
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
    def __init__(self, parent, width, height, brush=12, bg='black', fg='white', **kw):
        super().__init__(parent, bg='#ffffff', **kw)
        self.w = width
        self.h = height
        self.brush = brush
        self.bg = bg
        self.fg = fg
        self._on_draw_callback = None

        self.image = Image.new('L', (width, height), color=0 if bg == 'black' else 255)
        self.draw_ctx = ImageDraw.Draw(self.image)

        self.photo = ImageTk.PhotoImage(self.image)
        self.label = tk.Label(self, image=self.photo, bg='#f0f0f0', cursor='cross',
                              relief=tk.GROOVE, bd=2)
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
    # ── Brighter color palette ──
    BG = '#f0f4f8'           # light blue-gray
    CARD = '#ffffff'         # white card
    ACCENT = '#4f6ef7'       # vibrant blue
    ACCENT_HOVER = '#3b54d4' # darker blue
    SUCCESS = '#10b981'      # emerald green
    SUCCESS_HOVER = '#059669'
    WARNING = '#f59e0b'      # amber
    WARNING_HOVER = '#d97706'
    TEXT_PRIMARY = '#1e293b'
    TEXT_SECONDARY = '#64748b'
    TEXT_MUTED = '#94a3b8'
    BORDER = '#e2e8f0'
    CANVAS_BORDER = '#cbd5e1'

    def __init__(self, root):
        self.root = root
        self.root.title("MNIST 手写数字识别 — 交互测试")
        self.root.geometry("960x780")
        self.root.configure(bg=self.BG)
        self.root.minsize(900, 680)

        # ── Style ──
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=self.BG, borderwidth=0)
        style.configure('TNotebook.Tab', padding=[28, 10], font=('Microsoft YaHei', 11),
                        background='#dde4ed', foreground=self.TEXT_SECONDARY, borderwidth=0)
        style.map('TNotebook.Tab',
                  background=[('selected', self.CARD)],
                  foreground=[('selected', self.ACCENT)])
        style.configure('TCombobox', font=('Microsoft YaHei', 11),
                        fieldbackground='white', background='white',
                        arrowcolor=self.ACCENT)

        # ── Top bar: model selector ──
        self._build_topbar()

        # ── Notebook ──
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=(4, 16))

        self.tab1 = tk.Frame(self.notebook, bg=self.BG)
        self.tab2 = tk.Frame(self.notebook, bg=self.BG)
        self.notebook.add(self.tab1, text='  单个数字  ')
        self.notebook.add(self.tab2, text='  数字串  ')

        self._build_single()
        self._build_multi()

    # ═══ Top bar with model selector ════════════════════
    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=self.CARD)
        bar.pack(fill=tk.X, padx=16, pady=(14, 0))

        inner = tk.Frame(bar, bg=self.CARD)
        inner.pack(fill=tk.X, padx=20, pady=12)

        tk.Label(inner, text='MNIST 手写数字识别', font=('Microsoft YaHei', 16, 'bold'),
                 fg=self.ACCENT, bg=self.CARD).pack(side=tk.LEFT)

        # Model selector on right side
        right_frame = tk.Frame(inner, bg=self.CARD)
        right_frame.pack(side=tk.RIGHT)

        tk.Label(right_frame, text='模型选择  ', font=('Microsoft YaHei', 11),
                 fg=self.TEXT_SECONDARY, bg=self.CARD).pack(side=tk.LEFT)

        model_list = model_mgr.get_model_list()
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(right_frame, textvariable=self.model_var,
                                        values=model_list, state='readonly', width=32)
        self.model_combo.pack(side=tk.LEFT, padx=(4, 8))
        if model_list:
            self.model_combo.current(len(model_list) - 1)  # select the newest by default

        self.model_btn = tk.Button(right_frame, text='加载', font=('Microsoft YaHei', 10, 'bold'),
                                   bg=self.ACCENT, fg='white', relief=tk.FLAT,
                                   cursor='hand2', activebackground=self.ACCENT_HOVER,
                                   activeforeground='white', padx=14, pady=4,
                                   command=self._on_model_change)
        self.model_btn.pack(side=tk.LEFT)

        self.model_status = tk.Label(inner, text='', font=('Microsoft YaHei', 9),
                                     fg=self.TEXT_MUTED, bg=self.CARD)
        self.model_status.pack(side=tk.RIGHT, padx=(0, 16))

    def _on_model_change(self):
        selected = self.model_var.get()
        if not selected:
            return
        self.model_status.config(text='加载中...', fg=self.WARNING)
        self.root.update()
        ok = model_mgr.load_model(selected)
        if ok:
            info = model_mgr.get_info(selected)
            acc = info.get('acc', '?')
            if isinstance(acc, float):
                acc_str = f'{acc:.2f}%'
            else:
                acc_str = str(acc)
            self.model_status.config(text=f'✓ 准确率: {acc_str}', fg=self.SUCCESS)
        else:
            self.model_status.config(text='加载失败', fg='#ef4444')

    # ═══ Single Digit Tab ═══════════════════════════════
    def _build_single(self):
        mf = tk.Frame(self.tab1, bg=self.BG)
        mf.pack(fill=tk.BOTH, expand=True, padx=24, pady=20)

        # Left panel
        left = tk.Frame(mf, bg=self.CARD, relief=tk.FLAT, bd=0,
                        highlightbackground=self.BORDER, highlightthickness=1)
        left.pack(side=tk.LEFT, padx=(0, 20))
        left_inner = tk.Frame(left, bg=self.CARD)
        left_inner.pack(padx=20, pady=20)

        tk.Label(left_inner, text='✎ 用鼠标写数字', font=('Microsoft YaHei', 13, 'bold'),
                 fg=self.TEXT_PRIMARY, bg=self.CARD).pack(pady=(0, 10))
        self.pad1 = DrawPad(left_inner, 300, 300, brush=18, bg='black', fg='white')
        self.pad1.pack()
        self.pad1.set_draw_callback(self._predict_single)

        bf = tk.Frame(left_inner, bg=self.CARD)
        bf.pack(fill=tk.X, pady=(12, 0))
        tk.Button(bf, text='清除', font=('Microsoft YaHei', 11), bg='#e2e8f0', fg=self.TEXT_PRIMARY,
                  relief=tk.FLAT, cursor='hand2', activebackground='#cbd5e1',
                  activeforeground=self.TEXT_PRIMARY, padx=20, pady=6,
                  command=self._clear_single).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(bf, text='识别', font=('Microsoft YaHei', 11, 'bold'), bg=self.ACCENT, fg='white',
                  relief=tk.FLAT, cursor='hand2', activebackground=self.ACCENT_HOVER,
                  activeforeground='white', padx=20, pady=6,
                  command=self._predict_single).pack(side=tk.LEFT)

        # Right panel
        right = tk.Frame(mf, bg=self.CARD, relief=tk.FLAT, bd=0,
                         highlightbackground=self.BORDER, highlightthickness=1)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right, text='识别结果', font=('Microsoft YaHei', 14, 'bold'),
                 fg=self.TEXT_PRIMARY, bg=self.CARD).pack(pady=(24, 2))
        self.single_digit = tk.Label(right, text='?', font=('Arial', 72, 'bold'),
                                     fg=self.ACCENT, bg=self.CARD)
        self.single_digit.pack(pady=6)
        self.single_conf = tk.Label(right, text='在左侧画布上写一个数字',
                                    font=('Microsoft YaHei', 11), fg=self.TEXT_MUTED, bg=self.CARD)
        self.single_conf.pack()

        # Top-10 bars
        tk.Label(right, text='Top-10 概率', font=('Microsoft YaHei', 11, 'bold'),
                 fg=self.TEXT_SECONDARY, bg=self.CARD).pack(pady=(18, 6))
        self.bars_frame = tk.Frame(right, bg=self.CARD)
        self.bars_frame.pack(fill=tk.X, padx=32)
        self.s_bars = []
        for i in range(10):
            row = tk.Frame(self.bars_frame, bg=self.CARD)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=str(i), font=('Arial', 9, 'bold'), fg=self.TEXT_SECONDARY,
                     bg=self.CARD, width=2, anchor='e').pack(side=tk.LEFT, padx=(0, 8))
            bar = tk.Frame(row, bg=self.ACCENT, width=0, height=16)
            bar.pack(side=tk.LEFT); bar.pack_propagate(False)
            pct = tk.Label(row, text='', font=('Arial', 8), fg=self.TEXT_SECONDARY,
                           bg=self.CARD, width=7, anchor='w')
            pct.pack(side=tk.LEFT, padx=6)
            self.s_bars.append((bar, pct))

    def _clear_single(self):
        self.pad1.clear()
        self.single_digit.config(text='?')
        self.single_conf.config(text='在左侧画布上写一个数字')
        for bar, pct in self.s_bars:
            bar.config(width=0); pct.config(text='')

    def _predict_single(self, event=None):
        if model_mgr.model is None:
            self.single_conf.config(text='请先选择并加载模型')
            return
        img = self.pad1.get_image()
        pred, probs = predict_digit(img)
        self.single_digit.config(text=str(pred))
        self.single_conf.config(text=f'置信度: {probs[pred]*100:.1f}%')
        max_w = 260
        for digit, prob in enumerate(probs):
            bar, pct = self.s_bars[digit]
            bar.config(bg=self.ACCENT if digit == pred else '#dde4ed',
                       width=int(prob * max_w))
            pct.config(text=f'{prob*100:.0f}%' if prob > 0.01 else '')

    # ═══ Multi-Digit Tab ═══════════════════════════════
    def _build_multi(self):
        mf = tk.Frame(self.tab2, bg=self.BG)
        mf.pack(fill=tk.BOTH, expand=True, padx=24, pady=20)

        # Top card
        top_card = tk.Frame(mf, bg=self.CARD, relief=tk.FLAT, bd=0,
                            highlightbackground=self.BORDER, highlightthickness=1)
        top_card.pack(fill=tk.X)

        tk.Label(top_card, text='✎ 写一串数字（数字之间请保持一定间距）',
                 font=('Microsoft YaHei', 13, 'bold'), fg=self.TEXT_PRIMARY,
                 bg=self.CARD).pack(pady=(18, 10))
        self.pad2 = DrawPad(top_card, 720, 180, brush=14, bg='black', fg='white')
        self.pad2.pack(padx=20, pady=(0, 14))
        bf = tk.Frame(top_card, bg=self.CARD)
        bf.pack(fill=tk.X, padx=20, pady=(0, 16))
        tk.Button(bf, text='清除', font=('Microsoft YaHei', 11), bg='#e2e8f0', fg=self.TEXT_PRIMARY,
                  relief=tk.FLAT, cursor='hand2', activebackground='#cbd5e1',
                  activeforeground=self.TEXT_PRIMARY, padx=20, pady=6,
                  command=self._clear_multi).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(bf, text='识别数字串', font=('Microsoft YaHei', 11, 'bold'), bg=self.WARNING,
                  fg='white', relief=tk.FLAT, cursor='hand2', activebackground=self.WARNING_HOVER,
                  activeforeground='white', padx=20, pady=6,
                  command=self._predict_multi).pack(side=tk.LEFT)

        # Bottom card
        bottom = tk.Frame(mf, bg=self.CARD, relief=tk.FLAT, bd=0,
                          highlightbackground=self.BORDER, highlightthickness=1)
        bottom.pack(fill=tk.BOTH, expand=True, pady=(14, 0))

        tk.Label(bottom, text='识别结果', font=('Microsoft YaHei', 13, 'bold'),
                 fg=self.TEXT_PRIMARY, bg=self.CARD).pack(pady=(16, 2))
        self.multi_result = tk.Label(bottom, text='--', font=('Arial', 44, 'bold'),
                                     fg=self.WARNING, bg=self.CARD)
        self.multi_result.pack(pady=2)
        self.multi_hint = tk.Label(bottom, text='点击"识别数字串"按钮进行识别',
                                   font=('Microsoft YaHei', 10), fg=self.TEXT_MUTED, bg=self.CARD)
        self.multi_hint.pack()
        self.seg_frame = tk.Frame(bottom, bg=self.CARD)
        self.seg_frame.pack(pady=10)

    def _clear_multi(self):
        self.pad2.clear()
        self.multi_result.config(text='--')
        self.multi_hint.config(text='点击"识别数字串"按钮进行识别')
        for w in self.seg_frame.winfo_children():
            w.destroy()

    def _predict_multi(self):
        if model_mgr.model is None:
            self.multi_hint.config(text='请先选择并加载模型')
            return
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
            sf = tk.Frame(self.seg_frame, bg='#f8fafc', relief=tk.GROOVE, bd=1)
            sf.pack(side=tk.LEFT, padx=6, pady=4)
            seg_p = seg.resize((36, 50), Image.LANCZOS)
            photo = ImageTk.PhotoImage(seg_p)
            il = tk.Label(sf, image=photo, bg='#f8fafc')
            il.image = photo
            il.pack(pady=(4, 1))
            tk.Label(sf, text=f'→ {pred}', font=('Arial', 13, 'bold'),
                     fg=self.ACCENT, bg='#f8fafc').pack(pady=(0, 4))
        result_str = ''.join(results)
        self.multi_result.config(text=result_str)
        self.multi_hint.config(text=f'检测到 {len(results)} 个数字')
        print(f"Multi-digit: {len(results)} segments → {result_str}")


def main():
    try:
        root = tk.Tk()
        root.title("MNIST 手写数字识别")
        root.geometry("500x180")
        root.configure(bg='#f0f4f8')

        status_label = tk.Label(root, text="正在扫描模型...", font=('Microsoft YaHei', 14),
                                fg='#4f6ef7', bg='#f0f4f8')
        status_label.pack(expand=True)
        root.update()

        # Scan for available models
        available = model_mgr.get_model_list()
        if not available:
            import tkinter.messagebox as mb
            mb.showerror("模型未找到",
                         "未找到任何模型文件。\n\n请先训练模型:\n  python cnn_train_gpu.py\n\n"
                         "模型文件应位于 results/ 目录下。")
            root.destroy()
            sys.exit(1)

        # Pre-load the newest model
        print(f"Found {len(available)} model(s): {available}")
        newest = available[-1]
        print(f"Loading: {newest}")
        model_mgr.load_model(newest)
        print(f"Device: {model_mgr.device}")

        root.destroy()

        # Build full window
        root = tk.Tk()
        app = DigitRecognizerApp(root)
        # Auto-load info for pre-selected model
        sel = app.model_var.get()
        if sel:
            info = model_mgr.get_info(sel)
            acc = info.get('acc', '?')
            if isinstance(acc, float):
                acc_str = f'{acc:.2f}%'
            else:
                acc_str = str(acc)
            app.model_status.config(text=f'✓ 准确率: {acc_str}', fg=app.SUCCESS)

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

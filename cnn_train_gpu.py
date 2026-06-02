#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN MNIST training — GPU-accelerated version.
Auto-detects: CUDA (NVIDIA) > MPS (Apple M-series) > CPU fallback.
Uses mixed-precision (AMP) on CUDA for additional speed.

Usage:
    python -u cnn_train_gpu.py
"""
import os as _os
_os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.datasets import MNIST
import os
import sys
import time
import json
from datetime import datetime


def log_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


class MNIST_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.fc(x)


def main():
    # ── Device detection ────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        use_amp = True
        device_info = f'CUDA: {gpu_name} ({gpu_mem:.1f} GB)'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        use_amp = False
        device_info = 'MPS (Apple Metal)'
    else:
        device = torch.device('cpu')
        use_amp = False
        device_info = 'CPU (no GPU detected)'

    # ── Config ──────────────────────────────────────────
    gpu_mode = (device.type != 'cpu')
    BATCH_SIZE = 64 if gpu_mode else 128
    EPOCHS = 25 if gpu_mode else 15
    LEARNING_RATE = 0.001
    NUM_WORKERS = 2 if gpu_mode else 0

    DATA_DIR = 'mnist'
    RESULT_DIR = 'results'
    os.makedirs(RESULT_DIR, exist_ok=True)

    log_print('=' * 60)
    log_print(f'Device: {device_info}')
    log_print(f'AMP (mixed precision): {use_amp}')
    log_print(f'Batch size: {BATCH_SIZE}  Epochs: {EPOCHS}  LR: {LEARNING_RATE}')
    log_print(f'DataLoader workers: {NUM_WORKERS}')
    log_print('=' * 60)

    if not gpu_mode:
        log_print('WARNING: No GPU available. Install CUDA PyTorch for RTX acceleration:')
        log_print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124')
        log_print('')

    # ── Data loading ────────────────────────────────────
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = MNIST(DATA_DIR, train=True, transform=transform_train, download=True)
    test_set = MNIST(DATA_DIR, train=False, transform=transform_test, download=True)
    train_loader = data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=gpu_mode,
    )
    test_loader = data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=gpu_mode,
    )

    log_print(f'Train samples: {len(train_set)}, Test samples: {len(test_set)}')

    # ── Training state ──────────────────────────────────
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler('cuda') if use_amp else None

    total_params = sum(p.numel() for p in model.parameters())
    log_print(f'Parameters: {total_params:,}')

    # ── Metrics log ─────────────────────────────────────
    log = {
        'config': {
            'batch_size': BATCH_SIZE, 'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE, 'device': device_info,
            'amp': use_amp, 'optimizer': 'Adam',
            'scheduler': 'StepLR(step=5, gamma=0.5)',
            'loss_fn': 'CrossEntropyLoss',
            'train_samples': len(train_set), 'test_samples': len(test_set),
            'params': total_params,
        },
        'epochs': [],
    }

    best_acc = 0.0
    total_start = time.time()

    log_print(f'\nTraining started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log_print('=' * 60)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=gpu_mode)
            labels = labels.to(device, non_blocking=gpu_mode)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

            if batch_idx % 20 == 0:
                acc = 100. * train_correct / train_total
                log_print(f'  Epoch {epoch:2d} [{batch_idx:3d}/{len(train_loader):3d}] '
                          f'Loss: {loss.item():.4f}  Acc: {acc:.2f}%')

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # ── Test ──
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device, non_blocking=gpu_mode)
                labels = labels.to(device, non_blocking=gpu_mode)
                if use_amp:
                    with autocast('cuda'):
                        outputs = model(imgs)
                        test_loss += criterion(outputs, labels).item()
                else:
                    outputs = model(imgs)
                    test_loss += criterion(outputs, labels).item()
                _, preds = outputs.max(1)
                test_correct += preds.eq(labels).sum().item()
                test_total += labels.size(0)

        test_loss /= len(test_loader)
        test_acc = 100. * test_correct / test_total
        epoch_time = time.time() - epoch_start

        scheduler.step()

        entry = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 4),
            'test_loss': round(test_loss, 6),
            'test_acc': round(test_acc, 4),
            'epoch_time': round(epoch_time, 2),
        }
        log['epochs'].append(entry)

        star = ' ★' if test_acc > best_acc else ''
        best_acc = max(best_acc, test_acc)
        speed = f'{train_total / epoch_time:.0f} samples/s' if gpu_mode else ''
        log_print(f'  ── Epoch {epoch:2d} ── '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
                  f'Time: {epoch_time:.1f}s {speed}{star}')

    total_time = time.time() - total_start
    log['total_time'] = round(total_time, 2)
    log['best_acc'] = round(best_acc, 4)
    log['finished_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    log_print('=' * 60)
    log_print(f'Training finished.  Total: {total_time:.1f}s  Best: {best_acc:.2f}%')
    log_print(f'Device used: {device_info}')
    log_print('=' * 60)

    # ── Save model & log ────────────────────────────────
    torch.save(model.state_dict(), os.path.join(RESULT_DIR, 'mnist_cnn_gpu_v2.pth'))
    with open(os.path.join(RESULT_DIR, 'training_log_gpu_v2.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    log_print(f'Model saved to results/mnist_cnn_gpu_v2.pth')
    log_print(f'Log saved to results/training_log_gpu_v2.json')


if __name__ == '__main__':
    main()

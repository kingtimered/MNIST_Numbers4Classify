#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os as _os
_os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import json
from datetime import datetime

# ── Config ──────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'mnist'
RESULT_DIR = 'results'
os.makedirs(RESULT_DIR, exist_ok=True)

def log_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)

log_print(f'Device: {DEVICE}')
log_print(f'Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}')


# ── CNN Model ───────────────────────────────────────────
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
            nn.MaxPool2d(2),       # 28→14
            nn.Dropout2d(0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 14→7
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


# ── Data loading ────────────────────────────────────────
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
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

log_print(f'Train samples: {len(train_set)}, Test samples: {len(test_set)}')

# ── Training state ──────────────────────────────────────
model = MNIST_CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log_print(f'Parameters: {trainable_params:,} trainable / {total_params:,} total')

# ── Metrics log ─────────────────────────────────────────
log = {
    'config': {
        'batch_size': BATCH_SIZE, 'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE, 'device': str(DEVICE),
        'optimizer': 'Adam', 'scheduler': 'StepLR(step=5, gamma=0.5)',
        'loss_fn': 'CrossEntropyLoss',
        'train_samples': len(train_set), 'test_samples': len(test_set),
        'params': total_params,
    },
    'epochs': [],
}

best_acc = 0.0
total_start = time.time()

log_print('\n' + '=' * 60)
log_print('Training started at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
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
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = outputs.max(1)
        train_correct += preds.eq(labels).sum().item()
        train_total += labels.size(0)

        if batch_idx % 100 == 0:
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
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
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
    log_print(f'  ── Epoch {epoch:2d} ── '
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
          f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
          f'Time: {epoch_time:.1f}s{star}')

total_time = time.time() - total_start
log['total_time'] = round(total_time, 2)
log['best_acc'] = round(best_acc, 4)
log['finished_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

log_print('=' * 60)
log_print(f'Training finished.  Total: {total_time:.1f}s  Best: {best_acc:.2f}%')
log_print('=' * 60)

# ── Save model & log ────────────────────────────────────
torch.save(model.state_dict(), os.path.join(RESULT_DIR, 'mnist_cnn.pth'))
with open(os.path.join(RESULT_DIR, 'training_log.json'), 'w') as f:
    json.dump(log, f, indent=2, ensure_ascii=False)

# ── Plot curves ─────────────────────────────────────────
epochs_list = [e['epoch'] for e in log['epochs']]
train_losses = [e['train_loss'] for e in log['epochs']]
test_losses = [e['test_loss'] for e in log['epochs']]
train_accs = [e['train_acc'] for e in log['epochs']]
test_accs = [e['test_acc'] for e in log['epochs']]
lrs = [e['lr'] for e in log['epochs']]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(epochs_list, train_losses, 'b-o', label='Train Loss', markersize=6)
axes[0].plot(epochs_list, test_losses, 'r-o', label='Test Loss', markersize=6)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves'); axes[0].legend(); axes[0].grid(True)

axes[1].plot(epochs_list, train_accs, 'b-o', label='Train Acc', markersize=6)
axes[1].plot(epochs_list, test_accs, 'r-o', label='Test Acc', markersize=6)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Curves'); axes[1].legend(); axes[1].grid(True)

axes[2].plot(epochs_list, lrs, 'g-o', markersize=6)
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Learning Rate')
axes[2].set_title('Learning Rate Schedule'); axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
log_print('Curves saved to results/training_curves.png')

# ── Predict samples & plot ──────────────────────────────
model.eval()
classes = [str(i) for i in range(10)]
samples_per_class = 10
fig, axes = plt.subplots(10, samples_per_class, figsize=(16, 18))

with torch.no_grad():
    for class_idx in range(10):
        indices = (test_set.targets == class_idx).nonzero(as_tuple=True)[0][:samples_per_class]
        for j, idx in enumerate(indices):
            img, _ = test_set[idx]
            img_tensor = img.unsqueeze(0).to(DEVICE)
            pred = model(img_tensor).argmax(1).item()
            img_display = img.squeeze().cpu().numpy()
            ax = axes[class_idx, j]
            ax.imshow(img_display, cmap='gray')
            color = 'green' if pred == class_idx else 'red'
            ax.set_title(f'Pred:{pred}', fontsize=8, color=color)
            ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'predictions.png'), dpi=150, bbox_inches='tight')
plt.close()
log_print('Predictions grid saved to results/predictions.png')

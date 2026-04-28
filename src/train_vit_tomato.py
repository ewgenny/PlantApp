# src/train_vit_tomato.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import get_dataloaders

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from tqdm import tqdm

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

train_loader, val_loader, test_loader, class_names = get_dataloaders(
    root_dir="data/tomato",
    batch_size=16,
    num_workers=0,
    pin_memory=False
)

num_classes = len(class_names)

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=num_classes
).to(device)

train_labels = train_loader.dataset.targets

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

scheduler = CosineAnnealingLR(optimizer, T_max=15)

best_val_acc = 0
epochs = 15


def evaluate(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, pred = outputs.max(1)

            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    return 100 * correct / total


for epoch in range(epochs):
    model.train()

    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    train_acc = 100 * correct / total
    val_acc = evaluate(val_loader)

    print(f"Train {train_acc:.2f}% | Val {val_acc:.2f}%")

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names
        }, "models/vit_tomato_best.pth")

        print("Лучшая модель сохранена")

print("TEST")
test_acc = evaluate(test_loader)
print(f"Test Acc: {test_acc:.2f}%")

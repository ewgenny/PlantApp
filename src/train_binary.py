# src/train_binary.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import get_dataloaders

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import timm
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

train_loader, val_loader, test_loader, class_names = get_dataloaders(
    root_dir="data/binary",
    batch_size=16,
    num_workers=0,
    pin_memory=False
)

num_classes = len(class_names)
print("Классы:", class_names)

model = timm.create_model(
    "convnext_tiny",
    pretrained=True,
    num_classes=num_classes
).to(device)

for param in model.parameters():
    param.requires_grad = False

for param in model.head.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-4,
    weight_decay=0.01
)

scheduler = StepLR(optimizer, step_size=3, gamma=0.4)

best_val_acc = 0
epochs = 6


def evaluate(loader):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()

            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    return loss_sum / len(loader), 100 * correct / total


for epoch in range(epochs):
    model.train()

    correct = 0
    total = 0
    loss_sum = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    train_acc = 100 * correct / total
    train_loss = loss_sum / len(train_loader)

    val_loss, val_acc = evaluate(val_loader)

    print(
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.2f}% | "
        f"Val Acc {val_acc:.2f}%"
    )

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names
        }, "models/binary_convnext_best.pth")

        print("Лучшая модель сохранена")

print("TEST")
test_loss, test_acc = evaluate(test_loader)
print(f"Test Acc: {test_acc:.2f}%")
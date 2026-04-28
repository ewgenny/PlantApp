# src/evaluate_binary.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import get_dataloaders

import torch
import timm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, class_names = get_dataloaders(
    root_dir="data/binary",
    batch_size=32,
    num_workers=0,
    pin_memory=False
)

model = timm.create_model(
    "convnext_tiny",
    pretrained=False,
    num_classes=len(class_names)
).to(device)

checkpoint = torch.load(
    "models/binary_convnext_best.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names
))

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Binary Classifier Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()

plt.savefig("binary_confusion_matrix.png", dpi=300)

plt.show()
# src/evaluate_vit_cucumber.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import get_dataloaders

import torch
import timm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Устройство:", device)


train_loader, val_loader, test_loader, class_names = get_dataloaders(
    root_dir="data/cucumber",
    batch_size=16,
    num_workers=0,
    pin_memory=False
)

num_classes = len(class_names)


model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=num_classes
).to(device)

checkpoint = torch.load(
    "models/vit_cucumber_best.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, preds = outputs.max(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())


print("\n=== Classification Report ===")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

cm = confusion_matrix(y_true, y_pred)

print("=== Confusion Matrix ===")
print(cm)


plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix - ViT Cucumber")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
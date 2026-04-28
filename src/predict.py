# src/predict.py

import torch
import timm
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

plant_classes = ["cucumber", "tomato"]

binary_model = timm.create_model(
    "convnext_tiny",
    pretrained=False,
    num_classes=2
)

checkpoint = torch.load(
    "models/binary_convnext_best.pth",
    map_location=device
)

binary_model.load_state_dict(checkpoint["model_state_dict"])
binary_model.to(device)
binary_model.eval()

checkpoint = torch.load(
    "models/vit_tomato_best.pth",
    map_location=device
)

tomato_classes = checkpoint["class_names"]

vit_tomato = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=len(tomato_classes)
)

vit_tomato.load_state_dict(checkpoint["model_state_dict"])
vit_tomato.to(device)
vit_tomato.eval()

checkpoint = torch.load(
    "models/vit_cucumber_best.pth",
    map_location=device
)

cucumber_classes = checkpoint["class_names"]

vit_cucumber = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=len(cucumber_classes)
)

vit_cucumber.load_state_dict(checkpoint["model_state_dict"])
vit_cucumber.to(device)
vit_cucumber.eval()

def predict_single_tensor(image_tensor):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        binary_out = binary_model(image_tensor)
        binary_probs = torch.softmax(binary_out, dim=1)

        plant_idx = torch.argmax(binary_probs, dim=1).item()
        plant = plant_classes[plant_idx]

        if plant == "tomato":
            out = vit_tomato(image_tensor)
            probs = torch.softmax(out, dim=1)

            idx = torch.argmax(probs, dim=1).item()
            disease = tomato_classes[idx]

        else:
            out = vit_cucumber(image_tensor)
            probs = torch.softmax(out, dim=1)

            idx = torch.argmax(probs, dim=1).item()
            disease = cucumber_classes[idx]

        confidence = probs[0][idx].item()

    return plant, disease, confidence


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    return predict_single_tensor(image)


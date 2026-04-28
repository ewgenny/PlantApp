# src/utils.py

from PIL import Image
import torch
import timm
from preprocess import val_transform


def quick_predict(image_path, model_path, model_name="vit_base_patch16_224",
                  num_classes=2, device="cpu"):

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img = val_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return {
        "class": pred.item(),
        "confidence": conf.item(),
        "probs": probs.squeeze().cpu().tolist()
    }
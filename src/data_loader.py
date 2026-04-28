# src/data_loader.py

import os
from torchvision import datasets
from torch.utils.data import DataLoader
from preprocess import train_transform, val_transform


def get_dataloaders(root_dir, batch_size=16, num_workers=0, pin_memory=False):
    """
    Ожидаем структуру:

    root_dir/
        train/
        val/
        test/

    внутри каждой папки папки-классы
    """

    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names
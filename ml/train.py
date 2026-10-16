
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from plant_dataset import PlantDataset


def main():
    print("=== CHLOROMNESIS ===", flush=True)

    project_dir = Path(__file__).resolve().parent.parent
    split_file = project_dir / "data" / "splits" / "chloromnesis_split.json"
    data_dir = project_dir / "data" / "plantvillage"

    print("Ucitavanje podataka", flush=True)

    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    class_to_idx = split_data["class_to_idx"]

    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()

    train_dataset = PlantDataset(
        root_dir=data_dir,
        image_paths=split_data["train_paths"],
        class_to_idx=class_to_idx,
        transform=transform
    )

    val_dataset = PlantDataset(
        root_dir=data_dir,
        image_paths=split_data["val_paths"],
        class_to_idx=class_to_idx,
        transform=transform
    )

    print(f"Train: {len(train_dataset)} fotografija", flush=True)
    print(f"Validation: {len(val_dataset)} fotografija", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Uredjaj: {device}", flush=True)

    print("Ucitavanje MobileNetV3", flush=True)

    model = mobilenet_v3_small(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    num_classes = len(class_to_idx)

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )

    print(f"Broj klasa: {num_classes}", flush=True)

    print("Prebacivanje modela na GPU", flush=True)

    model = model.to(device)
    model.eval()

    print("Model spreman!", flush=True)

    
    print("Ucitavanje batch-a", flush=True)

    images, labels = next(iter(train_loader))
    images = images.to(device)

    print("Ulaz:", images.shape, flush=True)

    
    print("Pokretanje predikcije", flush=True)

    with torch.inference_mode():
        outputs = model(images)

    print("Izlaz:", outputs.shape, flush=True)
    print("Uredjaj izlaza:", outputs.device, flush=True)

    print("test uspesan", flush=True)


if __name__ == "__main__":
    main()

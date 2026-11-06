
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


    # Funkcija koja meri gresku modela
    criterion = nn.CrossEntropyLoss()

    # Optimizator azurira samo parametre koji nisu zamrznuti
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=0.001
    )

    print("\n Prvi trening ", flush=True)
    model.train()

    #ne menja se ponasanje zamrznutog feature extractora
    model.features.eval()

    total_loss=0.0
    correct=0
    total=0

    for batch_idx, (images,labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        #resetovanje gradijenta
        optimizer.zero_grad()

        #forward pass
        outputs=model(images)

        #racunanje greske
        loss=criterion(outputs, labels)

        #backpropagation
        loss.backward()

        #azuriranje parametara modela
        optimizer.step()

        #statistika
        total_loss+=loss.item() * images.size(0)
        predictions=outputs.argmax(dim=1)

        correct+=(predictions==labels).sum().item()
        total+=labels.size(0)

        if(batch_idx+1)%25==0:
            print(
                f"Batch {batch_idx+1}/{len(train_loader)} | "
                f"Loss: {total_loss/total:.4f} | "
                f"Accuracy: {correct/total*100:.2f}%",
                flush=True
            )
    
    train_loss=total_loss/total
    train_accuracy=correct/total*100

    print("\n Rezultati treninga: ")
    print(f"Loss: {train_loss:.4f}")
    print(f"Accuracy: {train_accuracy:.2f}%")

    print("\n Validacija: ", flush=True)

    model.eval()

    val_loss=0.0
    val_correct=0
    val_total=0

    with torch.inference_mode():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs=model(images)
            loss=criterion(outputs,labels)

            val_loss+=loss.item()*images.size(0)

            predictions=outputs.argmax(dim=1)
            val_correct+=(predictions==labels).sum().item()
            val_total+=labels.size(0)

    val_loss/=val_total
    val_accuracy=val_correct/val_total*100

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.2f}%")

    models_dir=project_dir/"models"
    models_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path=models_dir/"chloromnesis_v1.pth"

    torch.save({
        "model_state_dict" : model.state_dict(),
        "class_to_idx": class_to_idx,
        "epoch" : 1,
        "val_accuracy":val_accuracy
    }, checkpoint_path)

    print("Model sacuvan:", checkpoint_path)

if __name__ == "__main__":
    main()

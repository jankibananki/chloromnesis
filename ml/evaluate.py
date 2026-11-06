import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from plant_dataset import PlantDataset

def main():

    torch.set_num_threads(4)
    
    print("Evaluacija")

    project_dir=Path(__file__).resolve().parent.parent

    #ucitavanje prethodne podele podataka
    split_path=project_dir/"data"/"splits"/"chloromnesis_split.json"

    with open(split_path, "r", encoding="utf-8") as f:
        split_data=json.load(f)

    class_to_idx=split_data["class_to_idx"]
    num_classes=len(class_to_idx)

    #iste transformacije kao i tokom treninga
    weights=MobileNet_V3_Small_Weights.DEFAULT
    transform=weights.transforms()

    val_dataset=PlantDataset(
        root_dir=project_dir/"data"/"plantvillage",
        image_paths=split_data["val_paths"],
        class_to_idx=class_to_idx,
        transform=transform
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    #ucitavanje sacuvanog modela
    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model=mobilenet_v3_small(weights=None)

    model.classifier[-1]=nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )

    checkpoint_path = project_dir /"models"/"chloromnesis_v1.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model=model.to(device)
    model.eval()

    print("Model je uspesno ucitan.")
    print("Uredjaj: ", device)

    #prikupljane stvarnih oznaka i predikcija
    true_labels=[]
    predicted_labels=[]

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)

            outputs=model(images)
            predictions=outputs.argmax(dim=1)

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.cpu().tolist())

        #redosled kategorija mora da odgovara indeksima
        class_names=sorted(
            class_to_idx,
            key=lambda name: class_to_idx[name]
        )

        display_names = [
            name.replace("Tomato___", "").replace("_", " ")
            for name in class_names
        ]

        class_ids=list(range(num_classes))

        print("\n Classification report: ")

        print(classification_report(
            true_labels,
            predicted_labels,
            labels=class_ids,
            target_names=display_names,
            digits=3,
            zero_division=0
        ))

        print("\n Confusion matrix: ")

        matrix=confusion_matrix(
            true_labels,
            predicted_labels,
            labels=class_ids
        )

        for index, name in enumerate(display_names):
            print(f"{index}: {name}")

        print(matrix)
    

    # graficki prikaz confusion matrixa
    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=display_names
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d"
    )

    ax.set_title("Chloromnesis v1 - Confusion Matrix")

    fig.tight_layout()

    # cuvanje grafika radi dokumentacije
    reports_dir = project_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / "confusion_matrix_v1.png"

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Confusion matrix sacuvan:", output_path)


if __name__=="__main__":
    main()
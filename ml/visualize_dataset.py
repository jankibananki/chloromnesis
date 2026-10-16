from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

project_dir=Path(__file__).resolve().parent.parent


data_dir = project_dir / "data" / "plantvillage" / "raw" / "color"

classes = {
    "Tomato___healthy": "Healthy",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot"
}

random.seed(42)

fig, axes=plt.subplots(5,3,figsize=(10,14))

for row, (folder, label) in enumerate(classes.items()):
    class_dir=data_dir/folder

    image_files = sorted(class_dir.glob("*.JPG"))

    selected_images=random.sample(image_files,3)

    for col, image_path in enumerate(selected_images):
        image = Image.open(image_path).convert("RGB")

        axes[row, col].imshow(image)
        axes[row, col].set_title(f"{label} - {col+1}")
        axes[row, col].axis("off")


plt.tight_layout()
plt.show()
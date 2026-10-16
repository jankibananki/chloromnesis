from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter
from plant_dataset import PlantDataset
from torchvision import transforms
from torch.utils.data import DataLoader

project_dir=Path(__file__).resolve().parent.parent
splits_dir=project_dir / "data" / "splits"
splits_dir.mkdir(parents=True, exist_ok=True)

url="https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_train.txt"

train_file=splits_dir / "color_train.txt"

if not train_file.exists():
    print("Preuzimam listu...")
    urlretrieve(url,train_file)

print("Lista je spremna")

paths=train_file.read_text(encoding="utf-8").splitlines()

tomato_paths = [
    path for path in paths
    if path.startswith("raw/color/Tomato___")
]

class_counts = Counter(
    path.split("/")[2] for path in tomato_paths
)

print("Ukupno fotografija paradajza:", len(tomato_paths))

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

selected_classes=[
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot"
]

class_to_idx={
    name: index for index, name in enumerate(selected_classes)
}

selected_paths=[
    path for path in tomato_paths
    if path.split("/")[2] in class_to_idx
]

test_url = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_test.txt"

test_file=splits_dir / "color_test.txt"

if not test_file.exists():
    urlretrieve(test_url,test_file)

all_test_paths=test_file.read_text(
    encoding="utf-8"
).splitlines()

test_paths = [
    path for path in all_test_paths
    if path.split("/")[2] in class_to_idx
]

print("Train fotografije:", len(selected_paths))
print("Test fotografije:", len(test_paths))

train_set = set(selected_paths)
test_set = set(test_paths)

overlap = train_set.intersection(test_set)

print("Broj zajednickih fotografija:", len(overlap))

assert len(overlap) == 0, "Train i test se preklapaju!"

print("\nOdabrane kategorije:")

for name, index in class_to_idx.items():
    count=class_counts[name]
    percentage=count/len(selected_paths)*100
    print(f"{index}: {name} - {count} ({percentage:.1f}%)")

print("\nUkupno odabranih fotografija:", len(selected_paths))

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
]
)

dataset=PlantDataset(
    root_dir=project_dir / "data" / "plantvillage",
    image_paths=selected_paths,
    class_to_idx=class_to_idx,
    transform=transform
)

print("Broj fotografija:", len(dataset))

image,label=dataset[0]

image, label = dataset[0]

print("Tensor shape:", image.shape)
print("Tensor dtype:", image.dtype)
print("Label:", label)


loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(loader))

print("Batch images:", images.shape)
print("Batch labels:", labels.shape)
print("Labels:", labels.tolist())



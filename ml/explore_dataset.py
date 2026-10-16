from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter
from plant_dataset import PlantDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import json
from sklearn.model_selection import StratifiedGroupKFold

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

leaf_map_url = (
    "https://huggingface.co/datasets/mohanty/"
    "PlantVillage/resolve/main/leaf_grouping/leaf-map.json"
)

leaf_map_file = splits_dir / "leaf-map.json"

if not leaf_map_file.exists():
    print("Preuzimam leaf grouping metapodatke...")
    urlretrieve(leaf_map_url,leaf_map_file)

with open(leaf_map_file, "r", encoding="utf-8") as f:
    leaf_map = json.load(f)

def get_leaf_id(image_path):
    class_name=Path(image_path).parent.name
    identifier=Path(image_path).stem
    identifier=identifier.replace("_final_masked","")
    if "___" in identifier:
        identifier=identifier.split("___")[-1]

    identifier = identifier.split("copy")[0].strip().lower()

    suggestions=leaf_map.get(identifier,[])
    matches = [
        item for item in suggestions
        if item.startswith(class_name + ":::")
    ]

    if len(matches) != 1:
        return None
    
    return matches[0]


train_leaf_ids = [get_leaf_id(p) for p in selected_paths]
test_leaf_ids = [get_leaf_id(p) for p in test_paths]

print("\nLeaf ID provera:")
print("Train bez ID:", train_leaf_ids.count(None))
print("Test bez ID:", test_leaf_ids.count(None))

train_groups = set(train_leaf_ids) - {None}
test_groups = set(test_leaf_ids) - {None}

print("Jedinstvenih train listova:", len(train_groups))
print("Jedinstvenih test listova:", len(test_groups))
print("Zajednickih leaf_id:", len(train_groups & test_groups))

print("\nPrimer povezivanja:")
for path in selected_paths[:3]:
    print(Path(path).name, "->", get_leaf_id(path))


print("\nBroj identifikatora:", len(leaf_map))

example_key=next(iter(leaf_map))

print("Primer kljuca:", example_key)
print("Primer vrednosti:", leaf_map[example_key])

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

#dijagnosticka fja za proveru leaf id
def check_mapping(image_path):
    class_name=Path(image_path).parent.name
    identifier=Path(image_path).stem

    identifier=identifier.replace("_final_masked", "")

    if "___" in identifier:
        identifier = identifier.split("___")[-1]

    identifier=identifier.split("copy")[0].strip().lower()
    suggestions=leaf_map.get(identifier)

    if suggestions is None:
        return "Key not found", identifier

    matches = [
        item for item in suggestions
        if item.startswith(class_name + ":::")
    ] 

    if len(matches)==0:
        return "Class mismatch", identifier
    
    if len(matches)>1:
        return "Multiple matches", identifier
    
    return "OK", identifier

status_counts=Counter()
missing_examples=[]

for path in selected_paths:
    status, identifier = check_mapping(path)
    status_counts[status] += 1

    if status != "OK" and len(missing_examples)<5:
        missing_examples.append((Path(path).name, status, identifier))

print("\nMapping dijagnostika:")

for status, count in status_counts.items():
    print(f"{status}: {count}")

print("\nPrimeri problema:")

for filename, status, identifier in missing_examples:
    print(f"{filename} -> {status} -> {identifier}")


known_paths = [
    path for path in selected_paths
    if get_leaf_id(path) is not None
]

labels = [
    class_to_idx[Path(path).parent.name]
    for path in known_paths
]

groups = [
    get_leaf_id(path)
    for path in known_paths
]

print("Fotografije sa poznatim ID:", len(known_paths))


splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(known_paths, labels, groups)
)

train_paths = [known_paths[i] for i in train_idx]
val_paths = [known_paths[i] for i in val_idx]

train_groups = {groups[i] for i in train_idx}
val_groups = {groups[i] for i in val_idx}

print("Train:", len(train_paths))
print("Validation:", len(val_paths))
print("Zajednicki listovi:", len(train_groups & val_groups))

assert len(train_groups & val_groups) == 0


for name, paths in [
    ("TRAIN", train_paths),
    ("VALIDATION", val_paths)
]:
    counts = Counter(
        Path(path).parent.name for path in paths
    )

    print(f"\n{name}")

    for class_name in selected_classes:
        print(class_name, counts[class_name])

    assert all(counts[c] > 0 for c in selected_classes)



#cuvanje podele
split_data = {
    "train_paths": train_paths,
    "val_paths": val_paths,
    "class_to_idx": class_to_idx
}

split_file = splits_dir / "chloromnesis_split.json"

with open(split_file, "w", encoding="utf-8") as f:
    json.dump(split_data, f, indent=4)

print("Podela uspesno sacuvana!")



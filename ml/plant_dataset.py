from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class PlantDataset(Dataset):
    def __init__ (self,root_dir,image_paths,class_to_idx,transform=None):
        self.root_dir=Path(root_dir)
        self.image_paths=image_paths
        self.class_to_idx=class_to_idx
        self.transform=transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,index):
        image_path=self.image_paths[index]

        class_name = Path(image_path).parent.name
        label = self.class_to_idx[class_name]

        with Image.open(self.root_dir/image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image,label

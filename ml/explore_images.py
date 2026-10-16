import numpy as np
import torch

from pathlib import Path
from PIL import Image, ImageOps

project_dir = Path(__file__).resolve().parent.parent

image_path = project_dir / "data" / "leaf.jpg"

image = Image.open(image_path)

print("Dimenzije: ", image.size)
print("Rezim boja: ", image.mode)

image.show()

image = image.convert("RGB")

pixel=image.getpixel((200,200))

print("RGB vrednosti pixela: ", pixel)

image_array = np.array(image)

print("Oblik niza:", image_array.shape)
print("Tip podataka:", image_array.dtype)
print("Prvi piksel:", image_array[0,0])

normalized_array = image_array.astype(np.float32)/255.0

print("Tip nakon normalizacije:", normalized_array.dtype)
print("Prvi piksel nakon normalizacije:", normalized_array[0,0])

print("Najmanja vrednost:", normalized_array.min())
print("Najveca vrednost:", normalized_array.max())

target_size = (224,224)

resized_image = ImageOps.pad(
    image,
    target_size,
    method=Image.Resampling.LANCZOS,
    color=(0,0,0)
)

print("Nove dimenzije:", resized_image.size)
resized_image.show()

resized_array=np.array(resized_image)

resized_array=resized_array.astype(np.float32) / 255.0

print("Novi oblik:", resized_array.shape)
print("Novi tip:", resized_array.dtype)

image_tensor=torch.from_numpy(resized_array)

print("Tensor shape:", image_tensor.shape)
print("Tensor dtype:", image_tensor.dtype)

image_tensor=image_tensor.permute(2,0,1)
image_tensor=image_tensor.unsqueeze(0)
print("Final tensor shape:", image_tensor.shape)

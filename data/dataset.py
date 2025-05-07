from medmnist import PathMNIST
from torchvision.utils import save_image
from torchvision import transforms
from pathlib import Path
import torch

# Image transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Base output directory
base_dir = Path("data")

# Loop through each dataset split
for split in ['train', 'val', 'test']:
    dataset = PathMNIST(split=split, download=True)
    images, labels = dataset.imgs, dataset.labels

    for img, label in zip(images, labels):
        label = int(label[0])  # Extract integer label
        split_dir = base_dir / split / str(label)
        split_dir.mkdir(parents=True, exist_ok=True)

        img_tensor = transform(img)
        filename = f"{torch.randint(0, 1_000_000, (1,)).item()}.png"
        save_image(img_tensor, split_dir / filename)

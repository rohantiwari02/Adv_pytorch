import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medmnist import PathMNIST
from medmnist import INFO
from models.tiny_vgg import TinyVGG

def evaluate(model, val_loader, device='cpu'):
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            predicted = outputs.argmax(1)
            correct += (predicted == labels).sum().item()

    acc = correct / len(val_loader.dataset)
    print(f"Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28))
    ])
    
    val_dataset = PathMNIST(split='val', transform=transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    info = INFO['pathmnist']
    in_channels = info['n_channels']
    num_classes = len(info['label'])

    model = TinyVGG(in_channels, num_classes)
    model.load_state_dict(torch.load("models/model_1.pth", map_location=device))
    model.to(device)

    evaluate(model, val_loader, device)

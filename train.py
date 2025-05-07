import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from medmnist import PathMNIST
from medmnist import INFO

from models.tiny_vgg import TinyVGG  # make sure the class name matches

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28))
    ])

    # Get dataset info
    info = INFO['pathmnist']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    # Load datasets
    train_dataset = PathMNIST(split='train', transform=transform, download=True)
    val_dataset = PathMNIST(split='val', transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, n_channels, n_classes

def train(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Train Acc: {acc:.4f}")

    torch.save(model.state_dict(), "models/model_1.pth")
    print("Model saved to models/model_1.pth")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    train_loader, val_loader, in_channels, num_classes = get_dataloaders()

    model = TinyVGG(in_channels, num_classes)
    train(model, train_loader, val_loader, epochs=10, lr=1e-3, device=device)

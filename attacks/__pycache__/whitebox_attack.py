import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os



def attack(model, data_loader, epsilon):
    import torch.nn as nn
    loss_fn = nn.CrossEntropyLoss()
    correct = 0
    adv_examples = []
    model.eval()

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        init_preds = outputs.max(1, keepdim=True)[1]

        loss = loss_fn(outputs, labels)
        model.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        perturbed_data = fgsm_attack(images, epsilon, data_grad)

        final_outputs = model(perturbed_data)
        final_preds = final_outputs.max(1, keepdim=True)[1]

        correct += final_preds.eq(labels.view_as(final_preds)).sum().item()

        # Store adversarial examples for visualization
        if len(adv_examples) < 5:
            adv_examples.append((images[0].detach(), labels[0].detach(), perturbed_data[0].detach()))

    final_acc = correct / len(data_loader.dataset)
    print(f"FGSM Epsilon: {epsilon}\tTest Accuracy = {final_acc:.4f}")
    return final_acc, adv_examples
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os



def blackbox_fgsm_attack(surrogate_model, target_model, data_loader, epsilon, device):
    import torch.nn as nn
    loss_fn = nn.CrossEntropyLoss()
    correct_clean = 0
    correct_adv = 0
    adv_examples = []

    surrogate_model.eval()
    target_model.eval()

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        # Clean predictions from target model (before perturbation)
        with torch.no_grad():
            clean_outputs = target_model(images)
            clean_preds = clean_outputs.max(1)[1]
            correct_clean += clean_preds.eq(labels).sum().item()

        # Surrogate gradient
        outputs = surrogate_model(images)
        loss = loss_fn(outputs, labels)
        surrogate_model.zero_grad()
        loss.backward()
        data_grad = images.grad.data

        # Perturb
        perturbed_images = fgsm_attack(images, epsilon, data_grad)

        # Target model on adversarial examples
        with torch.no_grad():
            adv_outputs = target_model(perturbed_images)
            adv_preds = adv_outputs.max(1)[1]
            correct_adv += adv_preds.eq(labels).sum().item()

        # Collect visuals
        if len(adv_examples) < 5:
            for i in range(images.shape[0]):
                if len(adv_examples) >= 5:
                    break
                adv_examples.append((images[i].detach(), labels[i].detach(), perturbed_images[i].detach(), adv_preds[i].item()))

    total = len(data_loader.dataset)
    clean_acc = correct_clean / total
    adv_acc = correct_adv / total

    print(f"FGSM Epsilon: {epsilon:.2f}\tClean Accuracy = {clean_acc:.4f}\tAdversarial Accuracy = {adv_acc:.4f}")
    return clean_acc, adv_acc, adv_examples


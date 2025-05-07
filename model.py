import torch

state_dict = torch.load("best_model.pth", map_location="cpu")
print(state_dict.keys())

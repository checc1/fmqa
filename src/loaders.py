from torchvision import datasets, transforms
from torchvision.datasets import OxfordIIITPet, STL10, Caltech101
import torchvision.models as models
import torch.nn as nn
from torch.autograd import gradcheck
from warnings import filterwarnings
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.data import random_split


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = STL10(root='./data', download=False, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

split = int(0.7 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [split, len(dataset) - split])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

split_test = int(0.1 * len(train_dataset))
test_dataset, train_dataset = random_split(train_dataset, [split_test, len(train_dataset) - split_test])
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
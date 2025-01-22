import torch
import torch.nn as nn
import torchvision
import torchvision.models.vgg as vgg
from typing import Dict, List, Union
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn.functional as F

# parameter
learning_rate = 0.001
epochs = 50
batch_size = 16 # 512
train_size = 1000 # 60000

# dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
train_data = dsets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
print(train_data[0])
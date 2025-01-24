import torchvision.models.resnet as resnet
from typing import Any, Callable, List, Optional, Type, Union

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

import time

# parameter
learning_rate = 0.1
epochs = 1 # 150
batch_size = 64 # 256
train_size = 1000 # 60000
test_size = 100

# dataset
# train data
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_data = dsets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)

print(train_data.data.shape)

train_data_mean = train_data.data.mean(axis=(0,1,2)) / 255
train_data_std = train_data.data.std(axis=(0,1,2)) / 255

print(train_data_mean, train_data_std)
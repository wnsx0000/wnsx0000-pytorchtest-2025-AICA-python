import torch
import torchvision.datasets as dsets
import torchvision.transforms as transform
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SoftmaxModel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)
    def forward(self, x) :
        return self.linear(x)

# train
mnist_train = dsets.MNIST(root="MNIST_data/", train=True, transform=transform.ToTensor(), download=True)
mnist_train = Subset(mnist_train, list(range(60000)))
dataloader = DataLoader(mnist_train, batch_size=256, shuffle=True)
print(type(dataloader))
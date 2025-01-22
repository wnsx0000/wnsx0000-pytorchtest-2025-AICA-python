import torch
import torch.nn as nn

input = torch.Tensor(1, 3, 227, 227)
conv = nn.Conv2d(3, 1, 11, stride=4)
print(conv(input).shape)
pool = nn.MaxPool2d(10, stride=None, padding=0)


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
learning_rate = 0.005
epochs = 20 # 50
batch_size = 256 # 512
train_size = 10000 # 60000
test_size = 1000

# dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = dsets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
train_data = Subset(train_data, list(range(train_size)))
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = dsets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
test_data = Subset(test_data, list(range(test_size)))
test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

# model
cfgs: Dict[str, List[Union[str, int]]] = {
    "myconv11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "myconv16": [32, 32, 'M', 64, 64, 128, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M']
}
# inp 3, 32, 32
# M 32, 16, 16
# M 128, 8, 8
# M 512, 4, 4
# fc1
# fc2
# fc3
conv_layers = vgg.make_layers(cfgs["myconv16"], batch_norm=False)

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, 
                        init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096), # size를 4로 맞춰줌
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", 
                                                nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = VGG(conv_layers, num_classes=10)

# optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# train
train_start_time = time.time()
model.train()
for epoch in range(1, epochs + 1) :
    for dataloader_index, (data, label) in enumerate(train_dataloader) :
        # hypothesis
        hypothesis = model(data)

        # cost
        cost = F.cross_entropy(hypothesis, label)

        # train
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    print('Epoch {:4d}/{}, Cost: {:.6f}'.format(epoch, epochs, cost.item()))
train_end_time = time.time()
print("train time :" , (train_end_time - train_start_time), "s")

# test
test_start_time = time.time()
model.eval()
with torch.no_grad() :
    for dataloader_index, (data, label) in enumerate(test_dataloader) :
        hypothesis = model(data)
        prediction = hypothesis.argmax(dim=1).long()
        accuracy = (prediction == label).float().mean()

        cost = F.cross_entropy(hypothesis, label)

        print('test accuracy: {}, Cost: {:.6f}'.format(accuracy, cost.item()))
test_end_time = time.time()
print("train time :" , (test_end_time - test_start_time), "s")







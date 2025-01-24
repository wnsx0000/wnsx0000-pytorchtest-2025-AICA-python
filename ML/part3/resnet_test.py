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

import visdom
vis = visdom.Visdom()
graph_id = vis.line(Y=torch.zeros(1), opts=dict(title="train"), env="main")
vis.close(env="main")

import time

# parameter
learning_rate = 0.1
epochs = 100 # 150
batch_size = 256 # 256
train_size = 30000 # 60000
test_size = 3000

# dataset
# train data
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_data = dsets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)

train_data_mean = train_data.data.mean(axis=(0,1,2)) / 255
train_data_std = train_data.data.std(axis=(0,1,2)) / 255

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])
train_data = dsets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
train_data = Subset(train_data, list(range(train_size)))
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# test data
test_data = dsets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
test_data = Subset(test_data, list(range(test_size)))
test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

# model
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128 * block.expansion, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnet.Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, resnet.BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # input (b, 3, 32, 32)
        x = self.conv1(x) # (b, 16, 32, 32)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x) # (b, 64, 32, 32)
        x = self.layer2(x) # (b, 128, 32, 32)
        x = self.layer3(x) # (b, 256, 16, 16)
        x = self.layer4(x) # (b, 512, 8, 8)

        x = self.avgpool(x) # (b, 512, 1, 1)
        x = torch.flatten(x, 1) # (b, -1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 10, True)

# optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# train
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
train_start_time = time.time()
model.train()
vis_index = 1
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

        vis.line(X=torch.Tensor([vis_index]), Y=torch.Tensor([cost.item()]), win=graph_id, update="append", env="main")
        vis_index = vis_index + 1

    print('Epoch {:4d}/{}, Cost: {:.6f}'.format(epoch, epochs, cost.item()))
    lr_sche.step()
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







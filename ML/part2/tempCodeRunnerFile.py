import torch
import torchvision.datasets as dsets
import torchvision.transforms as transform
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MnistModel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    def forward(self, x) :
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x

# train
mnist_train = dsets.MNIST(root="MNIST_data/", train=True, transform=transform.ToTensor(), download=True)
mnist_train = Subset(mnist_train, list(range(10000)))
dataloader = DataLoader(mnist_train, batch_size=64, shuffle=True)

model = MnistModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 10
for epoch in range(1, epochs + 1) :
    for batch_index, batch_data in enumerate(dataloader) :
        x_train, y_train = batch_data
        x_train = x_train.view(-1, 28 * 28)
        
        # hypothesis
        hypothesis = model(x_train)

        # cost
        cost = F.cross_entropy(hypothesis, y_train)

        # train
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # if batch_index % 10 == 0 :
        #     print('Epoch {:4d}/{}, batch {}, Cost: {:.6f}'.format(epoch, epochs, batch_index, cost.item()))
    print('Epoch {:4d}/{}, Cost: {:.6f}'.format(epoch, epochs, cost.item()))
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transform
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

class MnistModel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 10)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)

        # weight initalization
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.xavier_uniform_(self.linear4.weight)
    def forward(self, x) :
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.linear4(x)
        
        return x

model = MnistModel()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# train
mnist_train = dsets.MNIST(root="MNIST_data/", train=True, transform=transform.ToTensor(), download=True)
mnist_train = Subset(mnist_train, list(range(20000)))
dataloader = DataLoader(mnist_train, batch_size=512, shuffle=True)

model.train()
train_start_time = time.time()
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

    print('Epoch {:4d}/{}, Cost: {:.6f}'.format(epoch, epochs, cost.item()))
train_end_time = time.time()
print("train time :" , (train_end_time - train_start_time), "s")

# test
mnist_test = dsets.MNIST(root="MNIST_data/", train=False, transform=transform.ToTensor(), download=True)
mnist_test = Subset(mnist_test, list(range(2000)))
dataloader = DataLoader(mnist_test, batch_size=len(mnist_test), shuffle=True)

model.eval()
with torch.no_grad() :
    for batch_index, batch_data in enumerate(dataloader) :
        x_test, y_test = batch_data
        x_test = x_test.view(-1, 28 * 28)

        # hypothesis
        hypothesis = model(x_test)
        prediction = hypothesis.argmax(dim=1).long()
        accuracy = (y_test == prediction).float().mean()

        # cost
        cost = F.cross_entropy(hypothesis, y_test)
        
        print('test accuracy: {}, Cost: {:.6f}'.format(accuracy, cost.item()))


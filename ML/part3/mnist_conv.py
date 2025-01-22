import torch
import torchvision.datasets as dsets
import torchvision.transforms as transform
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

# parameter
learning_rate = 0.001
batch_size = 128
epochs = 10
subset_size = 60000
test_size = 10000

# dataset
mnist_train = dsets.MNIST(root="MNIST_data/", train=True, transform=transform.ToTensor(), download=True)
mnist_train = Subset(mnist_train, list(range(subset_size)))
dataloader_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

mnist_test = dsets.MNIST(root="MNIST_data/", train=False, transform=transform.ToTensor(), download=True)
mnist_test = Subset(mnist_test, list(range(test_size)))
dataloader_test = DataLoader(mnist_test, batch_size=len(mnist_test), shuffle=True)

# model
class MnistModel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # |x| = (batch_size, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2) # |x| = (batch_size, 32, 14, 14)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # |x| = (batch_size, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2) # |x| = (batch_size, 64, 7, 7)
        )

        self.linear = nn.Linear(64 * 7 * 7, 10)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x) : # |x| = (batch_size, 1, 28, 28)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

model = MnistModel()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
model.train()
train_start_time = time.time()
for epoch in range(1, epochs + 1) :
    for batch_index, batch_data in enumerate(dataloader_train) :
        x_train, y_train = batch_data
        x_train = x_train.view(-1, 1, 28, 28)
        
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
model.eval()
with torch.no_grad() :
    for batch_index, batch_data in enumerate(dataloader_test) :
        x_test, y_test = batch_data
        x_test = x_test.view(-1, 1, 28, 28)

        # hypothesis
        hypothesis = model(x_test)
        prediction = hypothesis.argmax(dim=1).long()
        accuracy = (y_test == prediction).float().mean()

        # cost
        cost = F.cross_entropy(hypothesis, y_test)
        
        print('test accuracy: {}, Cost: {:.6f}'.format(accuracy, cost.item()))


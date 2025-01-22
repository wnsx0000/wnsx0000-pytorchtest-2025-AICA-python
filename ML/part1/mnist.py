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
model = SoftmaxModel()
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

# test
mnist_test = dsets.MNIST(root="MNIST_data/", train=False, transform=transform.ToTensor(), download=True)
mnist_test = Subset(mnist_test, list(range(2000)))
dataloader = DataLoader(mnist_test, batch_size=len(mnist_test), shuffle=True)

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


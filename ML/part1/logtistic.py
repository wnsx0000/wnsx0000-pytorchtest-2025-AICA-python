import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MyLogisticModel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x) :
        return self.sigmoid(self.linear(x))


x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] # (6, 2)
x_train = torch.FloatTensor(x_data)
y_data = [[0], [0], [0], [1], [1], [1]] # (6, 1)
y_train = torch.FloatTensor(y_data)

model = MyLogisticModel()

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(1, nb_epochs + 1) :
    # hypothesis
    hypothesis = model(x_train)

    # cost
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # learning
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{}\nHypothesis: {}, Cost: {:.6f}'.format(epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()))

x_test = torch.FloatTensor([[1, 2], [2, 3], [3, 1]])
y_test = torch.FloatTensor([[0], [0], [0]])

hypothesis = model(x_test)
prediction = hypothesis >= torch.FloatTensor([0.5])
rst = torch.mean((prediction == y_test).float())





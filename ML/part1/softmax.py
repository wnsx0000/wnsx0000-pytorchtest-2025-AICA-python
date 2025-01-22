import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

# 3개의 범주로 classification

class MySoftmaxModel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.linear = nn.Linear(4, 3)
    def forward(self, x) :
        return self.linear(x)

x_data = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
x_train = torch.FloatTensor(x_data) # (8, 4)
y_data = [2, 2, 2, 1, 1, 1, 0, 0]
y_train = torch.LongTensor(y_data) # (8,)

model = MySoftmaxModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000

for epoch in range(1, nb_epochs + 1) :
    # hypothesis
    hypothesis = model(x_train)

    # cost
    cost = F.cross_entropy(hypothesis, y_train)

    # learning
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print('Epoch {:4d}/{}\naccuracy: {}, Cost: {:.6f}'.format(epoch, nb_epochs, 
                                                                  (F.softmax(hypothesis).argmax(dim=1) == y_train).float().mean(), 
                                                                  cost.item()))


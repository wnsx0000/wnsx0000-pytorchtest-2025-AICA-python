# 2개의 layer를 사용하여 xor를 푸는 모델을 만들어보자

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class XorModel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x) :
        rst1 = self.sigmoid(self.linear1(x))
        rst2 = self.sigmoid(self.linear2(rst1))
        return rst2
    
x_train = torch.FloatTensor([[1, 1], [1, 0], [0, 1], [0, 0]])
y_train = torch.FloatTensor([[0], [1], [1], [0]])

model = XorModel()
optimizer = optim.SGD(model.parameters(), lr=1)
epochs = 5000

for epoch in range(1, epochs + 1) :
    # hypothesis
    hypothesis = model(x_train)

    # cost
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # learning
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    rst = ((hypothesis > 0.5).long() == y_train).float().mean()

    if (epoch % 500) == 0 :
        print('Epoch {:4d}/{}\nHypothesis: {}, Accuracy: {}, Cost: {:.6f}'.format(epoch, epochs, hypothesis.squeeze().detach(), rst, cost.item()))








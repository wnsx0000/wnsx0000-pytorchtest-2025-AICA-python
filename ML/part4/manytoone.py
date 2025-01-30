import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import visdom

import numpy as np


# parameter
batch_len = 10
data_dim = 5
hidden_dim = 10
output_dim = 1

learning_rate = 0.01
epochs = 500


# data
stock_data = np.loadtxt("./ML/part4/data-02-stock_daily.csv", delimiter=",")
stock_data = stock_data[::-1].copy() # reverse order

train_size = int(len(stock_data) * 0.7)
train_set = stock_data[0:train_size - 1] # 딱 나누어 떨어지도록 함
test_set = stock_data[train_size - 1:]

def minmax_norm(tensor) : # 2D tensor
    min = tensor.min(dim=0)[0]
    max = tensor.max(dim=0)[0]
    
    return ((tensor - min) / (max - min))
    
def set_xy_from_data(data_set) :
    x = data_set[:-1]
    y = data_set[1:, -1:]
    
    x = torch.FloatTensor(x) # (total, 5)
    y = torch.FloatTensor(y) # (total, 1)

    x = minmax_norm(x)
    y = minmax_norm(y)

    x = x.view(-1, batch_len, data_dim)
    y = y.view(-1, batch_len, output_dim)
    
    return x, y

x_train, y_train = set_xy_from_data(train_set)
x_test, y_test = set_xy_from_data(test_set)


# model
class StockDataModule(nn.Module) :
    def __init__(self, input_size, hidden_size, output_size, num_layers) :
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size, bias=True)

        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x) :
        x, status = self.rnn(x)
        x = self.fc(x)
        return x
    
model = StockDataModule(data_dim, hidden_dim, output_dim, num_layers=1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# train
model.train()
for epoch in range(1, epochs + 1) :
    # hypothesis
    hypothesis = model(x_train)

    # cost
    cost = F.mse_loss(hypothesis, y_train)

    # train
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % (epochs / 10) == 0 :
        print('Epoch {:4d}/{}, Cost: {:.6f}'.format(epoch, epochs, cost.item()))

# test
model.eval()

hypothesis = model(x_test)

vis = visdom.Visdom()
graph_id = vis.line(Y=torch.zeros(1), opts=dict(title="train"))
vis.close(env="main")

def printRst(tensor, graph_id) :
    tensor = tensor.view(-1)
    vis.line(X=torch.FloatTensor(list(range(len(tensor)))), Y=tensor, win=graph_id, update="append")    

printRst(y_test, graph_id)
printRst(hypothesis, graph_id)

# cost = F.mse_loss(hypothesis, y_test)

# accuracy_criteria = 0.1
# accuracy = (abs(hypothesis - cost) < accuracy_criteria).float().mean()

# print('Test, Cost: {:.6f}, accuracy: {}'.format(cost.item(), accuracy))




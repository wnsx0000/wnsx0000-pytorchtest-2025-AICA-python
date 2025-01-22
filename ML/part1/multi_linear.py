import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MyModel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x) :
        return self.linear(x) # 가설에 x를 대입하여 계산한 결과 반환

class MyDataset(Dataset) :
    def __init__(self) :
        self.x_data = [[73, 80, 75], 
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100]]
        self.y_data = [[152], [185], [180], [196]]
    def __len__(self) :
        return len(self.x_data)
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x_data)
        y = torch.FloatTensor(self.y_data)
        return x, y

model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=1e-5)
nb_epochs = 10

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for epoch in range(nb_epochs + 1):
    for batch_index, batch_data in enumerate(dataloader) :
            x_train, y_train = batch_data

            prediction = model(x_train)
            cost = F.mse_loss(prediction, y_train)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            print('Epoch {:4d}/{} batch_index: {}\nHypothesis: {}, Cost: {:.6f}'.format(epoch, nb_epochs, batch_index, prediction.squeeze().detach(), cost.item()))
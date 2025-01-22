import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# parameter
learning_rate = 0.00005
epochs = 7
batch_size = 8

# dataset 가공
# trans = transforms.Compose([
#     transforms.Resize((64, 128))
# ])
# train_data = torchvision.datasets.ImageFolder(root="./PyTorch/custom_data/origin_data", transform=trans)

# for data_index, data_value in enumerate(train_data) :
#     data, label = data_value
#     if label == 0 :
#         data.save("./PyTorch/custom_data/train_data/gray/%d_%d.jpeg" % (data_index, label))
#     else :
#         data.save("./PyTorch/custom_data/train_data/red/%d_%d.jpeg" % (data_index, label))

# dataset
trans = transforms.Compose([
    transforms.ToTensor()
])
train_data = torchvision.datasets.ImageFolder(root="./PyTorch/custom_data/train_data", transform=trans)
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# model
class ChairModel(nn.Module) :
    def __init__(self) :
        super().__init__()

        self.layer1 = nn.Sequential(
            # (-1, 3, 64, 128)
            nn.Conv2d(3, 6, 5),
            # (-1, 6, 60, 124)
            nn.ReLU(),
            nn.MaxPool2d(2)
            # (-1, 6, 30, 62)
        )
        self.layer2 = nn.Sequential(
            # (-1, 6, 30, 62)
            nn.Conv2d(6, 16, 5),
            # (-1, 16, 26, 58)
            nn.ReLU(),
            nn.MaxPool2d(2)
            # (-1, 16, 13, 29)
        )
        self.layer3 = nn.Sequential(
            # (-1, 16 * 13 * 29)
            nn.Linear(16 * 13 * 29, 120),
            # (-1, 120)
            nn.ReLU(),
            nn.Linear(120, 2)
            # (-1, 2)
        )

    def forward(self, x) :
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1) # flattern layer
        x = self.layer3(x)
        return x
model = ChairModel()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
model.train()
for epoch in range(1, epochs + 1) :
    for data_index, data_value in enumerate(dataloader) :
        data, label = data_value

        # hypothesis
        hypothesis = model(data)

        # cost
        cost = F.cross_entropy(hypothesis, label)

        # train
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    print('Epoch {:4d}/{}, Cost: {:.6f}'.format(epoch, epochs, cost.item()))

# test
model.eval()
trans = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor()
])
test_data = torchvision.datasets.ImageFolder(root="./PyTorch/custom_data/test_data", transform=trans)
dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)
with torch.no_grad() :
    for data_index, (data, label) in enumerate(dataloader) :
        # hypothesis
        hypothesis = model(data)
        prediction = hypothesis.argmax(dim=1).long()
        accuracy = (prediction == label).float().mean()

        # cost
        cost = F.cross_entropy(hypothesis, label)

        print('test accuracy: {}, Cost: {:.6f}'.format(accuracy, cost.item()))


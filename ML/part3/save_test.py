import torch
import torch.nn as nn

class MnistModel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # |x| = (batch_size, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2), # |x| = (batch_size, 32, 14, 14)
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
torch.save(model.state_dict(), "./model/mnistmodel.pth") # 저장

new_model = MnistModel()
new_model.load_state_dict(torch.load("./model/mnistmodel.pth")) # 불러오기



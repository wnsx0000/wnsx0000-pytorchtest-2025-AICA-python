import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

sentence = ("if you want to build a ship, don't drum up people together to "
"collect wood and don't assign them tasks and work, but rather "
"teach them to long for the endless immensity of the sea.")

epochs = 500
learning_rate = 0.5

# data
sequence_length = 10
char_list = list(set(sentence))
char_dic = {d:i for i, d in enumerate(char_list)}
dic_size = len(char_dic)

# data setting
x_data = []
y_data = []
for i in range(0, len(sentence) - sequence_length) :
    x_str = sentence[i : i + sequence_length]
    y_str = sentence[i + 1 : i + sequence_length + 1]
    x_data.append([char_dic[c] for c in x_str]) # x_data list에 추가
    y_data.append([char_dic[c] for c in y_str]) # y_data list에 추가

x_train = torch.LongTensor(x_data)
y_train = torch.LongTensor(y_data)

eye_matrix = torch.eye(dic_size)
x_one_hot = F.one_hot(x_train, num_classes=dic_size).float()

# model
class LongCharModel(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers) :
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x) :
        x, status = self.rnn(x)
        x = self.fc(x)
        return x

model = LongCharModel(dic_size, dic_size, 2)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1) :
    # hypothesis
    hypothesis = model(x_one_hot)
    
    # cost
    cost = F.cross_entropy(hypothesis.view(-1, dic_size), y_train.view(-1))

    # train
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # rst
    prediction = hypothesis.argmax(dim=2)
    rst_str = "" + sentence[0]
    for index, str in enumerate(prediction) :
        if index == 0 :
            rst_str += ''.join([char_list[n] for n in str])
        else :
            rst_str += ''.join([char_list[n] for n in str])[-1]

    if epoch % 10 == 0 :
        print('Epoch {:4d}/{}, Cost: {:.6f}\nExpected string: {}'
              .format(epoch, epochs, cost.item(), rst_str))

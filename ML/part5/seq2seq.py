import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


# parameter
learning_rate = 0.01
epochs = 100

hidden_size = 16


# data
def preprocess(raw) :
    x_set = set()
    y_set = set()

    for (i, o) in raw :
        x_set = x_set.union(set(i.split()))
        y_set = y_set.union(set(o.split()))

    x_list = list(x_set)
    y_list = list(y_set)

    x_dic = {d:i for i, d in enumerate(x_list)}
    y_dic = {d:i for i, d in enumerate(y_list)}

    x_data = []
    y_data = []
    for (i, o) in raw :
        x_data.append(torch.LongTensor([x_dic[d] for d in i.split()]))
        y_data.append(torch.LongTensor([y_dic[d] for d in o.split()]))

    x_train = nn.utils.rnn.pad_sequence(x_data, batch_first=True)
    y_train = nn.utils.rnn.pad_sequence(y_data, batch_first=True)
    
    return len(x_dic), len(y_dic), x_train, y_train

raw = [("I feel hungry.", "나는 배가 고프다."),
       ("Pytorch is very easy.", "파이토치는 매우 쉽다."),
       ("Pytorch is a framework for deep learning.", "파이토치는 딥러닝을 위한 프레임워크이다."),
       ("Pytorch is very clear to use.", "파이토치는 사용하기 매우 직관적이다.")]

x_dic_len, y_dic_len, x_train, y_train = preprocess(raw)
input_size = len(x_train)
output_size = len(y_train)


# model
class MyEncoder(nn.Module) :
    def __init__(self, num_embedding, hidden_size) :
        super().__init__()
        self.embed = nn.Embedding(num_embedding, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x) :
        x = self.embed(x) # (batch(문장 개수), input_size, hidden_size)
        _, hidden = self.gru(x)
        return hidden

class MyDecoder(nn.Module) :
    def __init__(self, num_embedding, hidden_size, output_label_size) :
        super().__init__()
        self.embed = nn.Embedding(num_embedding, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_label_size)

        self.output_label_size = output_label_size

    def forward(self, y, hidden_state) :
        y = self.embed(y)

        sos = torch.zeros(y.size(0), 1, y.size(2))
        y = y[:, :-1, :]
        y = torch.cat((sos, y), dim=1)

        x, _ = self.gru(y, hidden_state)

        x = self.linear(x)
        x = x.view(-1, x.size(2))

        return x

class Seq2Seq(nn.Module) :
    def __init__(
            self, 
            incoder_embedding, 
            decoder_embedding, 
            hidden_size, 
            input_size
            ) :
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = MyEncoder(incoder_embedding, hidden_size)
        self.decoder = MyDecoder(decoder_embedding, hidden_size, decoder_embedding)

    def forward(self, x, y) :
        hidden = self.encoder(x)
        x = self.decoder(y, hidden)
        return x

# train
model = Seq2Seq(x_dic_len, y_dic_len, hidden_size, input_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()

for epoch in range(1, epochs + 1) :
    # hypothesis
    hypothesis = model(x_train, y_train)

    # cost
    cost = F.cross_entropy(hypothesis, y_train.view(-1))

    # train
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    expection = hypothesis.argmax(dim=1)
    accuracy = (expection == y_train.view(-1)).float().mean()

    if epoch % (epochs / 10) == 0 :
        print('Epoch {:4d}/{}, Cost: {:.6f}, Accuracy: {}'.format(epoch, epochs, cost.item(), accuracy))    





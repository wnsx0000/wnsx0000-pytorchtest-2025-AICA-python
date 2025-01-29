import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

learning_rate = 0.1
epochs = 200

sample = "wow it's amazing!"

# data
data_list = list(set(sample)) # data에서 유일한 문자들의 list 추출
data_dic = {d:i for i, d in enumerate(data_list)} # 문자 별로 인덱스 지정

data_dic_rev = {i:d for i, d in enumerate(data_list)}

input_size = len(data_dic)
hidden_size = len(data_dic)

data = [data_dic[c] for c in sample] # data_src를 숫자로 변환
x_train = torch.FloatTensor(data[:-1])
y_train = torch.LongTensor(data[1:])

eye_matrix = torch.eye(len(data_list))
x_one_hot = eye_matrix[x_train.long() - 1]

# model
rnn = nn.RNN(input_size, hidden_size)

optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
for epoch in range(1, epochs + 1) :
    # hypothesis
    hypothesis, status = rnn(x_one_hot)
    
    # cost
    cost = F.cross_entropy(hypothesis, y_train)

    # train
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # rst
    prediction = hypothesis.argmax(dim=1)
    rst_str = sample[0] + ''.join([data_list[n] for n in prediction])
    print('Epoch {:4d}/{}, Cost: {:.6f}, Expected string: {}'.format(epoch, epochs, cost.item(), rst_str))





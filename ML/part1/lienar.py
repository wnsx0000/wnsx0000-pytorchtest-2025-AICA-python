import torch
import torch.optim as optim

input_data = torch.FloatTensor([[1], [2], [3]])
output_data = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W, b], lr=0.01)

iter_epoch = 1000
for epoch in range(1, iter_epoch + 1) :
    hypothesis = W * input_data + b
    cost = torch.mean((hypothesis - output_data) ** 2) # MSE        
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print(W, b)
print(W * 10 + b)



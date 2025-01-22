import visdom
import torch

x = torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
x = torch.transpose(x, 0, 1)
y = torch.Tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
y = torch.transpose(y, 0, 1)

x = torch.Tensor([1, 2, 3, 4, 5])
y = torch.Tensor([3, 2, 1, 2, 2])

vis = visdom.Visdom()
vis.line(X=x, Y=y, env="main")


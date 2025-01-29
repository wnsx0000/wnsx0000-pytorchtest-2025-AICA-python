import torch
import torch.nn as nn

x = torch.Tensor( # (2, 2, 3)
    [[[1, 2, 3], 
    [3, 4, 5]],

    [[2, 4, 6], 
    [6, 8, 10]]]
    )

linear = nn.Linear(2, 3)
print(linear(x))




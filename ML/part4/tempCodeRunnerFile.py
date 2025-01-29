import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

sentence = ("if you want to build a ship, don't drum up people together to "
"collect wood and don't assign them tasks and work, but rather "
"teach them to long for the endless immensity of the sea.")
print(len(sentence))
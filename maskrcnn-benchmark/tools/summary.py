import torch
import torch.nn as nn
from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleConv().to(device)

summary(model, [(1, 16, 16), (1, 28, 28)])
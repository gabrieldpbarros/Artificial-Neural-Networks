import os
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

# Unimos diversas transformações simultaneamente pela classe Compose
# data_transform = Compose([
#     
# ])
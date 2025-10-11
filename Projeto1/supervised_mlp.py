import os
import torch
import data

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose

class LinearModel(nn.Module):
    def __init__(self, in_features=8, out_features=1):
        """
        in_features: Número de features que o dataset possui
        out_features: Classificação binária
        """
        super(LinearModel, self).__init__() # instancia o nn.Module
        self.l1 = nn.Linear(in_features, out_features)
        self.sigmoid = nn.Sigmoid

    def forward(self, x):
        # 1. Passa os dados para a camada linear
        # v = Σ(wx) + b
        x = self.l1(x)
        # 2. 
        return x

# Definimos o uso de CUDA para o treinamento do modelo
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)

# Criação do modelo
torch.manual_seed(42)
model = LinearModel().to(device)
print(model)
# df = data.viewData()
# print(df)



# Unimos diversas transformações simultaneamente pela classe Compose
# data_transform = Compose([
#     
# ])
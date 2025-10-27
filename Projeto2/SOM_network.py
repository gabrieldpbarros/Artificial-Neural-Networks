import os
import torch

from data import prepareData
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose
from typing import Tuple
from view_model import *

class SOM():
    """ Como o PyTorch NN não possui um modelo específico para a criação de uma rede SOM, vamos manipular tensores para isso. """
    def __init__(self, x: int, y: int, dim=8):
        self.x = x
        self.y = y
        self.dim = dim

        # Inicialização dos pesos (matriz de 8 dimensões)
        self.weights = torch.rand(self.y, self.x, self.dim)

        # Criação do grid
        ar_x = torch.arange(self.x)
        ar_y = torch.arange(self.y)
        xx, yy = torch.meshgrid(ar_x, ar_y, indexing="xy")
        self.grid = torch.stack([xx, yy], dim=2).float()

    def _calc_influence(self, bmu_coords: torch.Tensor, sigma: float):
        """ 
        Calcula a influência do BMU nos neurônios em volta.
        """
        # --- 1. Encontra as distâncias em 2D e encontra o numerador do expoente de h ---
        diff_vectors = self.grid - bmu_coords
        dist_squared = torch.sum(diff_vectors**2, dim=2)

        den = 2 * sigma**2 + 1e-9 # o termo 1e-9 evita divisão por zero

        # -- 2. Calcula as influências no grid. ---
        influence = torch.exp(-dist_squared / den)
        return influence.unsqueeze(-1)
    
    def get_dist(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Método auxiliar com o objetivo de encontrar as distâncias dos neurônios no tensor em relação aos features.
        """
        feats_view = feats.view(-1, 1, 1, self.dim)
        diff_vectors = self.weights - feats_view
        # Calculamos pelo módulo da distância
        dists = torch.norm(diff_vectors, p=2, dim=2)
        return dists
    
    def find_bmu(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Método de cálculo do BMU (Best Matching Unit) da rede SOM. Recebe uma amostra de features e retorna o neurônio que melhor
        representa os dados.
        """
        # --- 1. Buscamos o neurônio vencedor, aplicando o módulo da distância ---
        dists = self.get_dist(feats)
        flat_bmu = torch.argmin(dists)

        # --- 2. Convertemos a saída do método argmin() para duas coordenadas ---
        # O método argmin achata o grid para um único número (numera o melhor neurônio), então manipulamos o valor para encontrar
        # as coordenadas correspondentes.
        bmu_x = flat_bmu % self.x
        bmu_y = flat_bmu // self.x

        return bmu_x, bmu_y

    def update_weights(self,
                       feats: torch.Tensor,
                       eta: float,
                       bmu_coords: torch.Tensor,
                       sigma: float) -> None:
        """
        Atualiza os pesos da rede.
        """
        # --- 1. Aplica o cálculo da influência ---
        inf = self._calc_influence(bmu_coords, sigma)

        # --- 2. Encontra a distância entre o feature e a localização do neurônio correspondente ---
        delta = feats.view(1, 1, self.dim) - self.weights

        # --- 3. Aplicação do cálculo final ---
        self.weights = self.weights + (eta * inf * delta)

DEVICE = "cuda"
EPOCHS = 50
ETA = 0.1
SIGMA = 5
TAL_RADIUS = 5
TAL_LR = 5
X_SIZE = 5
Y_SIZE = 5
PATH = os.path.join("db/","csgo_processed.csv")

def get_coords(index, pos, map_x):
    """
    Função auxiliar para o cálculo das coordenadas do BMU no erro topográfico.
    """
    flat = index[:, pos]
    bmu_x = flat % map_x
    bmu_y = flat // map_x

    return torch.stack([bmu_x, bmu_y], dim=1).float() 

def evaluateModel(model: SOM, loader: DataLoader, device=DEVICE) -> Tuple[float, float]:
    """
    Calcula o Erro de Quantização (QE) e o Erro Topográfico (TE) do modelo.

    ENTRADA:
        model: Modelo de rede neural.
        loader: DataLoader com os features.
        device: Dispositivo de aceleração.

    SAÍDA:
        Retorna uma tupla contendo, respectivamente, os erros topográfico e de quantização.
    """
    total_top_loss = 0.0
    total_quant_loss = 0.0
    samples = 0

    with torch.no_grad():
        for (features,) in loader:
            features = features.to(device)
            batch_size = features.size(0)
            samples += batch_size

            # --- 1. Calculamos as distâncias em cada amostra do lote ---
            dist_batches = model.get_dist(features)
            dist_flatten = dist_batches.view(batch_size, -1)

            # --- 2. Calculamos os dois erros --
            # Erro de Quantização
            min_dists, _ = torch.min(dist_flatten, dim=1)
            total_quant_loss += torch.sum(min_dists)
            # Erro Topográfico
            _, top2 = torch.topk(dist_flatten, 2, dim=1, largest=False) # retorna os dois menores elementos do tensor de distâncias
            x = model.x
            bmu1 = get_coords(top2, 0, x)
            bmu2 = get_coords(top2, 1, x)
            # Encontramos as distâncias no grid
            grid_dist = torch.norm(bmu1 - bmu2, p=2, dim=1)
            total_top_loss += torch.sum(grid_dist > 1.0)
    avg_tp = (total_top_loss / samples).item()
    avg_qt = (total_quant_loss / samples).item()

    return avg_tp, avg_qt

def trainModel(model: SOM, 
               loader: DataLoader,
               epoch: int,
               eta: float=ETA,
               sigma: float=SIGMA,
               device=DEVICE) -> None:
    """
    Função de treinamento do modelo, percorrendo uma época.

    ENTRADA:
        model: Modelo de rede neural a ser treinado.
        loader: DataLoader de treinamento.
        epoch: Época atual do treinamento.
        eta: Taxa de aprendizado da rede.
        sigma: Raio de influência do BMU.
        device: Dispositivo de aceleração utilizado.
    """
    sigma_adjusted = sigma * np.exp(-epoch / TAL_RADIUS)
    eta_adjusted = eta * np.exp(-epoch / TAL_LR)

    # Itera sobre o dataloader (amostra por amostra)
    for (features_batch,) in loader:
        features_batch = features_batch.to(device)

        for features in features_batch:
            # --- 1. Encontra o BMU ---
            features = features.to(device)
            bmu_x, bmu_y = model.find_bmu(features)
            bmu_coords = torch.tensor([bmu_x, bmu_y], dtype=torch.float32, device=device)

            # --- 2. Atualiza os pesos ---
            model.update_weights(features, eta_adjusted, bmu_coords, sigma_adjusted)

def main():
    # Carregamos os dados localmente
    loader, labels = prepareData(PATH)
    first_batch = next(iter(loader))
    feats = first_batch[0]
    n_feats = feats.shape[1]

    # Definimos o uso de CUDA para o treinamento do modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # --- 1. Instanciação do modelo ---
    torch.manual_seed(42)
    model = SOM(X_SIZE, Y_SIZE, dim=n_feats)
    # Como a classe não é um nn.Module, precisamos mover os parâmetros para o dispositivo de acelaração
    model.weights = model.weights.to(device)
    model.grid = model.grid.to(device)

    # --- 2. Treinamento e validação do modelo ---
    # Armazenamos os erros seguindo o padrão {época: erro obtido}
    topographic_loss = {}
    quantization_loss = {}
    for i in range(EPOCHS):
        trainModel(model, loader, i, device=device)
        tp, qt = evaluateModel(model, loader, device)
        topographic_loss[i] = tp
        quantization_loss[i] = qt

    plotLosses(topographic_loss, quantization_loss, "Erro topográfico", "Erro de quantização")

if __name__ == "__main__":
    main()
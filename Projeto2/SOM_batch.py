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
        """
        PARÂMETROS:
            x: Quantidade de neurônios no eixo x do grid.
            y: Quantidade de neurônios no eixo y do grid.
            dim: Quantidade de features da amostra.
        """
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
        AGORA ACEITA UM LOTE DE COORDENADAS DE BMU.
        bmu_coords: Tensor de shape [BatchSize, 2] (contendo [x, y] para cada amostra)
        """
        batch_size = bmu_coords.size(0)
        
        # Prepara tensores para broadcasting:
        # grid: [Y, X, 2] -> [1, Y, X, 2]
        # bmu_coords: [B, 2] -> [B, 1, 1, 2]
        grid_view = self.grid.unsqueeze(0)
        bmu_view = bmu_coords.view(batch_size, 1, 1, 2)

        # --- 1. Encontra as distâncias em 2D para todo o lote ---
        # [B, Y, X, 2]
        diff_vectors = grid_view - bmu_view
        # [B, Y, X]
        dist_squared = torch.sum(diff_vectors**2, dim=3) # dim 3 é a dimensão [x, y]

        den = 2 * sigma**2 + 1e-9

        # -- 2. Calcula as influências no grid para todo o lote. ---
        # [B, Y, X]
        influence = torch.exp(-dist_squared / den)
        
        # Retorna com a dimensão extra para features: [B, Y, X, 1]
        return influence.unsqueeze(-1)
    
    def get_dist(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Método auxiliar com o objetivo de encontrar as distâncias dos neurônios no tensor em relação aos features.
        """
        feats_view = feats.view(-1, 1, 1, self.dim)
        diff_vectors = self.weights - feats_view
        # Calculamos pelo módulo (norma) da distância
        dists = torch.pow(diff_vectors, 2)
        dists_total = torch.sum(dists, dim=-1)
        return dists_total
    
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
                             feats_batch: torch.Tensor,
                             eta: float,
                             bmu_coords: torch.Tensor,
                             sigma: float) -> None:
        """
        Atualiza os pesos da rede usando um LOTE (BATCH) de dados.
        feats_batch: Tensor de features [BatchSize, 8]
        bmu_coords: Tensor de coordenadas dos BMUs [BatchSize, 2]
        """
        batch_size = feats_batch.size(0)
        
        # --- 1. Aplica o cálculo da influência (já em lote) ---
        # inf_batch tem shape [B, Y, X, 1]
        inf_batch = self._calc_influence(bmu_coords, sigma)

        # --- 2. Encontra o delta (x - W) para todo o lote ---
        # feats_view: [B, 1, 1, 8]
        # weights_view: [1, Y, X, 8]
        feats_view = feats_batch.view(batch_size, 1, 1, self.dim)
        weights_view = self.weights.unsqueeze(0)
        
        # delta_batch tem shape [B, Y, X, 8]
        delta_batch = feats_view - weights_view

        # --- 3. Calcula a atualização média ---
        # Multiplica a influência (h) pelo delta (x-W) para cada amostra
        # all_updates tem shape [B, Y, X, 8]
        all_updates = eta * inf_batch * delta_batch
        
        # Calcula a ATUALIZAÇÃO MÉDIA para o lote
        # avg_update tem shape [Y, X, 8]
        avg_update = torch.mean(all_updates, dim=0) # Média ao longo da dimensão do lote

        # --- 4. Aplicação do cálculo final ---
        self.weights = self.weights + avg_update

DEVICE = "cuda"
EPOCHS = 200
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

def trainModelBatch(model: SOM, 
                     loader: DataLoader,
                     epoch: int,
                     eta: float=ETA,
                     sigma: float=SIGMA,
                     device=DEVICE) -> None:
    """
    Função de treinamento do modelo (VERSÃO BATCH), percorrendo uma época.
    """
    # Decaimento (igual a antes)
    sigma_adjusted = sigma * np.exp(-epoch / TAL_RADIUS)
    eta_adjusted = eta * np.exp(-epoch / TAL_LR)

    # Itera sobre o dataloader (AGORA SÓ UM LOOP)
    for (features_batch,) in loader:
        features_batch = features_batch.to(device)
        batch_size = features_batch.size(0)

        # --- 1. Encontra os BMUs para o LOTE INTEIRO ---
        # dists_batch tem shape [B, Y, X] (com a correção em get_dist)
        dists_batch = model.get_dist(features_batch)
        
        # dist_flatten tem shape [B, Y*X]
        dist_flatten = dists_batch.view(batch_size, -1)
        
        # flat_bmu_indices tem shape [B] (um índice para cada amostra)
        flat_bmu_indices = torch.argmin(dist_flatten, dim=1)

        # Converte os índices planos para coordenadas (y, x)
        bmu_y = flat_bmu_indices // model.x
        bmu_x = flat_bmu_indices % model.x
        
        # bmu_coords tem shape [B, 2] (contendo [x, y], como o seu grid)
        bmu_coords = torch.stack([bmu_x, bmu_y], dim=1).float()

        # --- 2. Atualiza os pesos (em lote) ---
        model.update_weights(features_batch, eta_adjusted, bmu_coords, sigma_adjusted)

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
        trainModelBatch(model, loader, i, device=device)
        tp, qt = evaluateModel(model, loader, device)
        topographic_loss[i] = tp
        quantization_loss[i] = qt

    saving = os.path.join("assets/", "SOM_batch.png")

    plotLosses(topographic_loss, quantization_loss, "Erro topográfico", "Erro de quantização", saving)

if __name__ == "__main__":
    main()
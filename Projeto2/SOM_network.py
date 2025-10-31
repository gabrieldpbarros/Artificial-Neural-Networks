import os
import torch

from data import prepareData
import numpy as np
from torch.utils.data import DataLoader
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
        # Calculamos pelo módulo (norma) da distância
        dists = torch.norm(diff_vectors, p=2, dim=-1)
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
EPOCHS = 60
ETA = 0.1
SIGMA = 5
TAL_RADIUS = 200
TAL_LR = 200
X_SIZE = 20
Y_SIZE = 20
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

def generateHitMap(model: SOM, loader: DataLoader, device=DEVICE):
    """
    Função que gera um hitmap.
    """
    hit_map = torch.zeros((model.y, model.x), dtype=torch.int64, device=device)
    
    # 2. Desativa gradientes (como na avaliação)
    with torch.no_grad():
        for (features,) in loader:
            features = features.to(device)
            batch_size = features.size(0)

            # 3. Obtém distâncias e BMUs
            dist_batches = model.get_dist(features)
            dist_flatten = dist_batches.view(batch_size, -1)
            flat_bmu_indices = torch.argmin(dist_flatten, dim=1)
            
            # 4. Usa bincount para contar as ocorrências de cada índice de BMU
            bmu_counts = torch.bincount(flat_bmu_indices, minlength=model.x * model.y)
            
            # 5. Adiciona as contagens ao hit_map
            # (Reformulando de [Y*X] para [Y, X])
            hit_map += bmu_counts.view(model.y, model.x)

    # 6. Move para a CPU para plotar
    return hit_map.cpu().numpy()

def generateUMatrix(model: SOM, device=DEVICE):
    """
    Função que gera uma U-Matrix.
    """
    u_matrix = torch.zeros((model.y, model.x), dtype=torch.float32, device=device)
    
    # Pega os pesos (shape [Y, X, 8])
    weights = model.weights

    for y in range(model.y):
        for x in range(model.x):
            current_weight = weights[y, x]
            total_dist = 0.0
            neighbor_count = 0

            # Itera sobre os 8 vizinhos + o próprio (para simplificar os loops)
            for j in range(max(0, y - 1), min(model.y, y + 2)):
                for i in range(max(0, x - 1), min(model.x, x + 2)):
                    if (y == j and x == i):
                        continue # Não calcula a distância para si mesmo

                    # 1. Pega o peso do vizinho
                    neighbor_weight = weights[j, i]
                    # 2. Calcula a distância 8D
                    dist = torch.norm(current_weight - neighbor_weight, p=2)
                    total_dist += dist
                    neighbor_count += 1
            
            # 3. A U-Matrix armazena a distância MÉDIA aos vizinhos
            u_matrix[y, x] = total_dist / neighbor_count

    # 4. Move para a CPU e plota
    return u_matrix.cpu().detach().numpy()

def prepareHeatMap(model: SOM, features_names) -> Dict[str, np.ndarray]:
    weights_cpu = model.weights.cpu().detach().numpy() # Shape [Y, X, 8]
    
    plane_data_dict = {}
    for i in range(model.dim):
        feature_name = features_names[i]
        feature_plane = weights_cpu[:, :, i] # Pega o "plano" 2D [Y, X]
        plane_data_dict[feature_name] = feature_plane

    return plane_data_dict

def generateLabelMap(model: SOM, loader: DataLoader, labels: torch.Tensor, device=DEVICE):
    """
    Calcula a matriz de taxa de vitória (média de labels) para cada neurónio.
    
    ENTRADA:
        model: O seu modelo SOM treinado.
        loader: O DataLoader (apenas com features).
        labels: O tensor 1D completo de labels (JÁ NO DEVICE CORRETO).
        device: O dispositivo (ex: "cuda").

    SAÍDA:
        Uma matriz 2D Numpy com a taxa de vitória média por neurónio.
    """
    # Mapas para acumular valores (numerador e denominador)
    label_map = torch.zeros((model.y, model.x), dtype=torch.float32, device=device)
    hit_map = torch.zeros((model.y, model.x), dtype=torch.float32, device=device)
    
    label_idx = 0 # Índice para sincronizar os labels
    with torch.no_grad():
        for (features,) in loader:
            features = features.to(device)
            batch_size = features.size(0)

            # Pega o 'slice' de labels correspondente a este lote
            # (Assume que 'labels' já está no 'device')
            batch_labels = labels[label_idx : label_idx + batch_size]
            
            # --- 1. Encontra os BMUs do lote ---
            dist_batches = model.get_dist(features)
            dist_flatten = dist_batches.view(batch_size, -1)
            flat_bmu_indices = torch.argmin(dist_flatten, dim=1) # shape [B]

            # --- 2. Acumulação Eficiente ---
            # Acumula os labels (numerador)
            label_map.view(-1).scatter_add_(dim=0, index=flat_bmu_indices, src=batch_labels)
            
            # Acumula as contagens (denominador)
            hit_map.view(-1).scatter_add_(dim=0, index=flat_bmu_indices, src=torch.ones_like(batch_labels))

            # Avança o índice dos labels
            label_idx += batch_size

    # --- 3. Calcula a Média Final ---
    hit_map[hit_map == 0] = 1e-9 # Evita divisão por zero (para neurónios não visitados)
    win_rate_map = (label_map / hit_map)
    
    return win_rate_map.cpu().numpy()

def main():
    # Carregamos os dados localmente
    loader, labels = prepareData(PATH)
    first_batch = next(iter(loader))
    feats = first_batch[0]
    n_feats = feats.shape[1]
    features_names = [
        'ct_health', 't_health', 'ct_players_alive', 't_players_alive',
        'ct_equipment_value', 't_equipment_value', 'time_left', 'bomb_planted'
    ]

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
    
    losses_path = os.path.join("assets/", "SOM_simple.png")
    hitmap_path = os.path.join("assets/", "hitmap1.png")
    umatrix_path = os.path.join("assets/", "umatrix1.png")
    heatmap_path = os.path.join("assets/", "heatmaps1.png")
    labelheatmap_path = os.path.join("assets/", "labelheatmaps1.png")

    plotLosses(topographic_loss, quantization_loss, "Erro topográfico", "Erro de quantização", losses_path)
    hitMap(generateHitMap(model, loader, device=device), hitmap_path)
    uMatrix(generateUMatrix(model, device=device), umatrix_path)
    heatMap(prepareHeatMap(model, features_names), heatmap_path)
    labelHeatMap(generateLabelMap(model, loader, labels, device=device), labelheatmap_path)

if __name__ == "__main__":
    main()
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Dict

def plotLosses(loss1: Dict[int, float], loss2: Dict[int ,float], loss1_label: str, loss2_label: str, saving) -> None:
    plt.close("all")
    plt.figure()
    plt.plot(loss1.keys(), loss1.values(), label=loss1_label)
    plt.plot(loss2.keys(), loss2.values(), label=loss2_label)
    plt.title(f"{loss1_label} e {loss2_label}")
    plt.xlabel("Épocas")
    plt.ylabel("Erro")
    plt.legend()
    plt.savefig(saving)
    plt.show()

def hitMap(hit_map, saving) -> None:
    plt.close("all")
    plt.figure()
    sns.heatmap(hit_map, annot=True, fmt="d", cmap="viridis")
    plt.title("Hit Map (Frequência dos BMUs)")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.savefig(saving)
    plt.show()


def uMatrix(u_matrix, saving) -> None:
    plt.close("all")
    plt.figure()
    # Usar um mapa de cores reverso (ex: "gray_r") é comum para U-Matrix
    sns.heatmap(u_matrix, cmap="gray_r", annot=False)
    plt.title("U-Matrix (Fronteiras dos Clusters)")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.savefig(saving)
    plt.show()

def heatMap(plane_data: Dict[str, np.ndarray], saving) -> None:
    """
    Recebe um dicionário de planos de componentes (heatmaps) e plota
    num grid.
    
    plane_data: Dicionário no formato {nome_da_feature: array_2D_de_pesos}
    saving: Caminho para salvar a imagem (ex: "component_planes.png")
    """
    plt.close("all")
    num_features = len(plane_data)

    # --- 1. Determina o layout do grid (ex: 8 features -> 3x3 grid) ---
    grid_size = math.ceil(math.sqrt(num_features))
    
    # --- 2. Cria a figura e os eixos (subplots) ---
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 5))
    # Transforma axs num array 1D para fácil iteração
    axs = axs.flatten() 

    i = 0
    # --- 3. Itera sobre o dicionário e plota cada heatmap ---
    for feature_name, feature_plane in plane_data.items():
        sns.heatmap(feature_plane, ax=axs[i], cmap="coolwarm", cbar=True)
        axs[i].set_title(feature_name)
        axs[i].set_aspect('equal') # Garante que os "pixels" sejam quadrados
        i += 1
    
    # --- 4. Desliga os eixos (subplots) extras que não foram usados ---
    for j in range(i, len(axs)):
        axs[j].axis('off')

    plt.tight_layout() # Ajusta para evitar sobreposição de títulos
    plt.savefig(saving)
    plt.show()

def labelHeatMap(win_rate_map: np.ndarray, saving) -> None:
    """
    Plota um Heatmap de Taxa de Vitória (média de labels).
    
    ENTRADA:
        win_rate_map: Matriz 2D Numpy com valores (idealmente) entre 0 e 1.
        saving: Caminho para salvar a imagem (ex: "label_heatmap.png")
    """
    plt.close("all")
    plt.figure()
    # 'coolwarm' é ótimo: Azul (0, Vitória CT) -> Vermelho (1, Vitória T)
    # vmin=0 e vmax=1 fixam a escala de cores, o que é crucial
    sns.heatmap(win_rate_map, cmap="coolwarm", annot=False, vmin=0, vmax=1)
    plt.title("Heat Map de Taxa de Vitória (1.0 = Vitória T)")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.savefig(saving)
    plt.show()
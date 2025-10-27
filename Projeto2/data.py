import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torchvision.transforms import Compose
from typing import Tuple

def prepareData(path: str,
                batch_size: int = 32) -> Tuple[DataLoader, torch.Tensor]:
    """
    Recebe o caminho de um dataset e retorna os dados processados e preparados para serem utilizados
    no processo de aprendizado do modelo.

    ENTRADA:
        path: Caminho do arquivo.
        batch_size (opcional): Tamanho dos lotes nos dataloaders.
    SAÍDA:
        Tuple[DataLoader, torch.Tensor]: Tupla que contém os dados normalizados em um DataLoader,
        sem labels, e um tensor com as labels do dataset.
    """
    # --- 1. Carregamos o dataset completo, para uma divisão posterior ---
    full_data = pd.read_csv(path)

    # --- 2. Separação dos features e do label ---
    feats_df = full_data.iloc[:, :-1]
    label_df = full_data.iloc[:, -1]
    # Convertemos em tensores antes de normalizar
    feats_tensor = torch.tensor(feats_df.values, dtype=torch.float32)
    label_tensor = torch.tensor(label_df.values, dtype=torch.float32)

    # --- 3. Etapa de normalização dos dados ---
    # Usamos o método de normalização por distribuição
    # dim=0 indica que estamos trabalhando com as colunas
    mean = feats_tensor.mean(dim=0)
    std = feats_tensor.std(dim=0)
    # Evita divisão por zero se uma feature for constante
    std[std == 0] = 1.0
    feats_normalized = (feats_tensor - mean) / std

    # --- 4. Criação do TensorDataset e do DataLoader ---
    feats_dataset = TensorDataset(feats_normalized)
    data_loader = DataLoader(feats_dataset, batch_size=batch_size, shuffle=True)

    return data_loader, label_tensor

def filterData(path: str = 'db/') -> None:
    """
    Função específica para o dataset escolhido. Filtra os dados para manter apenas:

    'ct_health',
    't_health',
    'ct_money',
    't_money',
    'time_left',
    'bomb_planted',
    'ct_players_alive',
    't_players_alive',
    'round_winner'

    ENTRADA:
        path: Caminho do dataset original, com exceção do arquivo em si.
    SAÍDA:
        None
    """
    # Carrega o arquivo CSV grande.
    db_path = os.path.join(path, 'csgo_round_snapshots.csv')
    df = pd.read_csv(db_path)

    # --- 1. Convertendo a coluna alvo (label) ---
    # Mapeia 'CT' para 1 e 'T' para 0
    df['round_winner_numeric'] = df['round_winner'].map({'CT': 1, 'T': 0})

    # --- 2. Converte a coluna bomb_planted para inteiros ---
    # False se torna 0 e True se torna 1
    df['bomb_planted'] = df['bomb_planted'].astype(int)

    # --- 3. Selecionando e ordenando as colunas finais ---
    # Lista com as features selecionadas por você
    features = [
        'ct_health',
        't_health',
        'ct_money',
        't_money',
        'time_left',
        'bomb_planted',
        'ct_players_alive',
        't_players_alive'
    ]

    # Coluna do label (alvo)
    label = 'round_winner_numeric'

    # Cria o DataFrame final com as features e o label como a ÚLTIMA coluna
    # Isso é importante para que nossa classe NumericalDataset funcione corretamente
    final_df = df[features + [label]]
    # Retiramos os dados
    filter = final_df[final_df['ct_health'] == 500.0]
    filter = filter[filter['t_health'] == 500.0]
    final_df.drop(filter.index, inplace=True)

    # --- 4. Salvando o novo arquivo CSV ---
    output_filename = 'csgo_processed.csv'
    final_path = os.path.join(path, output_filename)
    final_df.to_csv(final_path, index=False)

def viewData(path: str = 'db/') -> pd.DataFrame:
    arch = 'csgo_processed.csv'
    file_path = os.path.join(path, arch)

    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":    
    filterData()
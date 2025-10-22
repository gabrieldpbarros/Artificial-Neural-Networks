import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torchvision.transforms import Compose
from typing import Tuple

class CustomDB(Dataset):
    """ Classe que adapta o database para o formato desejado pela NN """
    def __init__(self, file_path, transform=None, target_transform=None):
        df = pd.read_csv(file_path)

        # Assumimos que as labels estão na última coluna do dataframe, então separamos as features como
        # todos os valores anteriores à última coluna e os labels como a última coluna. Todos esses valores
        # são convertidos para ndarrays, facilitando a integração com os tensores
        features_np = df.iloc[:, :-1].values
        labels_np = df.iloc[:, -1].values

        self.features = torch.tensor(features_np, dtype=torch.float32)
        self.labels = torch.tensor(labels_np, dtype=torch.float32)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_sample = self.features[idx]
        label_sample = self.labels[idx]

        # Realiza as transformações, caso solicitadas
        if self.transform:
            feature_sample = self.transform(feature_sample)
        if self.target_transform:
            label_sample = self.target_transform(label_sample)

        return feature_sample, label_sample

def normalizeSubset(subset, mean, std):
    """ Função auxiliar para aplicar a normalização """
    features = torch.cat([subset[i][0].unsqueeze(0) for i in range(len(subset))], dim=0)
    labels = torch.tensor([subset[i][1] for i in range(len(subset))])
    
    # Normalização
    normalized_features = (features - mean) / std
    # Formato mais otimizado para armazenar o dataset pronto para ser carregado no dataloader
    return TensorDataset(normalized_features, labels.unsqueeze(1))

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
    full_data = CustomDB(file_path=path)

    # --- 2. Etapa de normalização dos dados ---
    # Usamos o método de normalização por distribuição
    # dim=0 indica que estamos trabalhando com as colunas
    mean = full_data.mean(dim=0)
    std = full_data.std(dim=0)
    # Evita divisão por zero se uma feature for constante
    std[std == 0] = 1.0

    data_normalized = normalizeSubset(full_data, mean, std)

    # Construimos os DataLoaders de cada conjunto
    data_loader = DataLoader(data_normalized, batch_size=batch_size, shuffle=True)

    return data_loader

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

filterData()
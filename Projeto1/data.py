import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split
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
        label_sample = self.featres[idx]

        # Realiza as transformações, caso solicitadas
        if self.transform:
            feature_sample = self.transform(feature_sample)
        if self.target_transform:
            label_sample = self.target_transform(label_sample)

        return feature_sample, label_sample
    
def prepareData(path: str, proportion: Tuple[float, float, float], data_transform: Compose = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Recebe o caminho de um dataset e retorna os dados processados e preparados para serem utilizados
    no processo de aprendizado do modelo.

    ENTRADA:
        path: Caminho do arquivo.
        proportion: Proporção em que os dados serão divididos para cada conjunto. São respectivamente os conjuntos de
        treino, validação eteste. Ex: (0.8, 0.1, 0.1)
        data_transform (opcional): Transformações que serão aplicadas nos dados para normalização.
    SAÍDA:
        Tuple[torch.tensor, torch.tensor, torch.tensor]: Tupla que contém, respectivamente, os dados de treino, validação
        e teste aleatorizados e normalizados.
    """
    # Carregamos o dataset completo, para uma divisão posterior
    full_data = CustomDB(
        file_path=path,
        transform=data_transform
    )

    # Calculamos o tamanho de cada conjunto
    dataset_size = len(full_data)
    train_size = int(dataset_size * proportion[0])
    val_size = int(dataset_size * proportion[1])
    test_size = dataset_size - train_size - val_size # garantimos que cobriremos o dataset completo

    # Geramos uma seed para a aleatorização dos dados
    generator = torch.Generator().manual_seed(42)
    # Dividimos os dados do dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_data,
        [train_size, val_size, test_size],
        generator=generator
    )

    # Construimos os DataLoaders de cada conjunto
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Não aleatorizamos os conjuntos de validação e de teste pois queremos uma avaliação consistente
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    # --- 2. Selecionando e ordenando as colunas finais ---
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

    # --- 3. Salvando o novo arquivo CSV ---
    output_filename = 'csgo_processed.csv'
    final_path = os.path.join(path, output_filename)
    final_df.to_csv(final_path, index=False)
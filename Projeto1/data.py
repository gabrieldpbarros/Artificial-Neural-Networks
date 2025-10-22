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

def normalize_subset(subset, mean, std):
    """ Função auxiliar para aplicar a normalização """
    features = torch.cat([subset[i][0].unsqueeze(0) for i in range(len(subset))], dim=0)
    labels = torch.tensor([subset[i][1] for i in range(len(subset))])
    
    # Normalização
    normalized_features = (features - mean) / std
    # Formato mais otimizado para armazenar o dataset pronto para ser carregado no dataloader
    return TensorDataset(normalized_features, labels.unsqueeze(1))

def prepareData(path: str,
                proportion: Tuple[float, float, float],
                batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Recebe o caminho de um dataset e retorna os dados processados e preparados para serem utilizados
    no processo de aprendizado do modelo.

    ENTRADA:
        path: Caminho do arquivo.
        proportion: Proporção em que os dados serão divididos para cada conjunto. São respectivamente os conjuntos de
        treino, validação eteste. Ex: (0.8, 0.1, 0.1)
        batch_size (opcional): Tamanho dos lotes nos dataloaders.
    SAÍDA:
        Tuple[torch.tensor, torch.tensor, torch.tensor]: Tupla que contém, respectivamente, os dados de treino, validação
        e teste aleatorizados e normalizados.
    """
    # --- 1. Carregamos o dataset completo, para uma divisão posterior ---
    full_data = CustomDB(file_path=path)

    # --- 2. Calculamos o tamanho de cada conjunto ---
    dataset_size = len(full_data)
    train_size = int(dataset_size * proportion[0])
    val_size = int(dataset_size * proportion[1])
    test_size = dataset_size - train_size - val_size # garantimos que cobriremos o dataset completo

    # Geramos uma seed para a aleatorização dos dados
    generator = torch.Generator().manual_seed(42)
    # --- 3. Dividimos os dados do dataset ---
    train_subset, val_subset, test_subset = random_split(
        full_data,
        [train_size, val_size, test_size],
        generator=generator
    )

    # --- 4. Etapa de normalização dos dados ---
    # Usamos o método de normalização por distribuição
    # dim=0 indica que estamos trabalhando com as colunas
    train_features = torch.cat([train_subset[i][0].unsqueeze(0) for i in range(len(train_subset))], dim=0)
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0)
    # Evita divisão por zero se uma feature for constante
    std[std == 0] = 1.0

    train_normalized = normalize_subset(train_subset, mean, std)
    val_normalized = normalize_subset(val_subset, mean, std)
    test_normalized = normalize_subset(test_subset, mean, std)

    # Construimos os DataLoaders de cada conjunto
    train_loader = DataLoader(train_normalized, batch_size=batch_size, shuffle=True)
    # Não aleatorizamos os conjuntos de validação e de teste pois queremos uma avaliação consistente
    val_loader = DataLoader(val_normalized, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_normalized, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

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
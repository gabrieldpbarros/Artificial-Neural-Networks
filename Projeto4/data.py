import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

def normalizeDataset(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    """
    Função auxiliar que aplica normalização no conjunto de dados.

    Args:
        df: DataFrame com os dados que serão utilizados.
        mean: Média do dataset de treino.
        std: Desvio padrão do dataset de treino.

    Output:
        Dataframe normalizado no formato pd.DataFrame.
    """
    return (df - mean) / std

def createSequences(data: np.ndarray, seq_len: int, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Função auxiliar que prepara os dados para serem inseridos no formato de
    série temporal no tensor conforme a janela temporal indicada.

    Args:
        data: Dados da série.
        seq_len: Tamanho da janela temporal.
        target_idx: Índice da coluna que utilizaremos para a predição.

    Output:
        Retorna a janela temporal e o valor de predição, ambos como ndarrays.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len]) # janela multivariada = X
        y.append(data[i + seq_len, target_idx])   # valor futuro = y

    return np.array(X), np.array(y)

def prepareData(path: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Carrega os dados em formato de um DataLoader customizado.

    Args:
        path: Caminho para o arquivo .csv que contém os dados.
        batch_size: Tamanho dos lotes de treinamento.

    Output:
        Retorna os dados separados conforme a proporção 75%, 15%, 15% como dataloaders
    """
    # --- 1. Montagem do DataFrame ---
    df = pd.read_csv(path)

    # --- 2. Normalização e separação dos conjuntos ---
    full_size = len(df)
    train_size = int(0.75 * full_size)
    val_size = int(0.15 * full_size)
    # Precisamos fazer essa inversão pois a série apresenta os valores mais recentes
    # nas primeiras linhas e os mais antigos nas últimas.
    df = df.iloc[::-1].reset_index(drop=True)
    # Esta etapa conserta o erro de geração de NaNs.
    df.loc[[0, 1], 'Gain'] = 0.0
    df.loc[[0, 1], 'Percent_Gain'] = 0.0

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:(train_size + val_size)]
    test_df = df.iloc[(train_size + val_size):]

    mean = train_df.mean(axis=0)
    std = train_df.std(axis=0)
    train_df = normalizeDataset(train_df, mean, std)
    val_df = normalizeDataset(val_df, mean, std)
    test_df = normalizeDataset(test_df, mean, std)

    # --- 3. Criação das janelas temporais
    seq_len = 5 # janela de 5 sequências
    target_col = "Avg_players"
    target_idx = df.columns.get_loc(target_col)

    train_np = train_df.to_numpy()
    # Extensão da janela dos conjuntos de validação e teste
    val_extended = pd.concat([train_df.tail(seq_len), val_df])
    test_extended = pd.concat([val_df.tail(seq_len), test_df])

    val_np = val_extended.to_numpy()
    test_np = test_extended.to_numpy()

    X_train_np, y_train_np = createSequences(train_np, seq_len, target_idx)
    X_val_np,   y_val_np   = createSequences(val_np, seq_len, target_idx)
    X_test_np,  y_test_np  = createSequences(test_np, seq_len, target_idx)
    # Remoção das janelas extras, evitando data leakage
    if len(X_val_np) > seq_len:
        X_val_np = X_val_np[seq_len:]
        y_val_np = y_val_np[seq_len:]

    if len(X_test_np) > seq_len:
        X_test_np = X_test_np[seq_len:]
        y_test_np = y_test_np[seq_len:]

    # --- 4. Conversão para tensor ---
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)

    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1)

    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_val, y_val)
    test_dataset  = TensorDataset(X_test, y_test)

    # --- 5. Criação dos dataloaders ---
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, mean, std

def filterData(path: str = "db/") -> None:
    """
    Como desejamos apenas os dados do jogo Warframe, devemos remover todos os
    dados relativos a outros jogos, bem como manter apenas as features relevantes
    para o treinamento da rede.

    Args:
        path: Caminho da pasta do dataset.
    """
    arc_path = os.path.join(path, "Valve_Player_Data.csv")
    df = pd.read_csv(arc_path)

    labels = [
        "Month_Year",
        "Date"
    ]

    features = [
        "Avg_players",
        "Gain",
        "Percent_Gain",
    ]

    # Conversão das porcentagens em decimal
    df["Percent_Gain"] = df["Percent_Gain"].str.rstrip("%").astype(float) / 100

    warframe_df = df[df["Game_Name"] == "Warframe"]
    warframe_labels = warframe_df[labels]
    warframe_features = warframe_df[features]

    output_path_labels = os.path.join(path, "Warframe_Player_Data_Labels.csv")
    output_path_features = os.path.join(path, "Warframe_Player_Data_Features.csv")
    warframe_labels.to_csv(output_path_labels, index=False)
    warframe_features.to_csv(output_path_features, index=False)

if __name__ == "__main__":
    filterData()
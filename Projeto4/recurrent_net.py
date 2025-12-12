import torch
import numpy as np
import pandas as pd

from data import prepareData
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple
from view_loss import plotLosses, plotSeries

class LSTM(nn.Module):
    def __init__(self, in_features: int = 3, hidden_size: int = 1, out_features: int = 1):
        """
        Args:
            in_features: Número de features que o dataset possui (recebemos os 3 features)
            out_features: Previsão do modelo (prevemos apenas a quantidade média de jogadores)
        """
        super(LSTM, self).__init__() # instancia o nn.Module
        self.lstm = nn.LSTM(in_features, hidden_size=hidden_size, batch_first=True)
        # Camada de mapeamento
        self.l1 = nn.Linear(hidden_size, out_features)

    def forward(self, input):
        """
        Args:
            input: (batch_size, seq_len, in_features)
        """
        out, (a, c) = self.lstm(input)
        a_last = a[-1]
        
        out = self.l1(a_last)

        return out
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 1000
BATCH_SIZE = 32
HIDDEN_SIZE = 64
ETA = 0.005
PATH = "db/Warframe_Player_Data_Features.csv"

def trainEpoch(model: nn.Module,
               train_loader: DataLoader, 
               optimizer,
               loss_function: callable,
               device=DEVICE) -> float:
    """
    Função de treinamento do modelo, percorrendo uma época.

    Args:
        model: Modelo de rede neural a ser treinado.
        train_loader: DataLoader de treinamento.
        optimizer: Método de cálculo da retropropagação.
        loss_function: Função de erro definida pelo usuário.
        device: Dispositivo de aceleração utilizado.
    Output:
        Retorna a média do erro calculado nas iterações da época.
    """
    model.train()
    total_loss = 0.0

    # Itera sobre o dataloader
    for features, labels in train_loader:
        # --- 1. Enviamos os dados para o dispositivo de aceleração ---
        features = features.to(device)
        labels = labels.to(device)
        # --- 2. Zeramos o gradiente da classe de gradiente estocástico ---
        optimizer.zero_grad()
        # --- 3. Feedforward ---
        # Passamos os features como x1, x2, ..., xn para o modelo gerar uma saída
        predict = model(features)
        # --- 4. Cálculo do erro da saída ---
        # Aplicação da função de erro
        out_loss = loss_function(predict, labels) # args: (saída do modelo, saída esperada)
        # --- 5. Retropropagação ---
        out_loss.backward()
        # --- 6. Atualização dos pesos com base na retropropagação ---
        optimizer.step()

        # Convertemos o erro (tensor) em um valor numérico
        total_loss += out_loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluateEpoch(model: nn.Module,
                  val_loader: DataLoader,
                  loss_function: callable,
                  device=DEVICE) -> float:
    """
    Função de validação do modelo, percorrendo uma época. Como não estamos treinando o modelo, não precisamos
    calcular o gradiente descendente estocástico, o qual é necessário para atualizar os pesos sinápticos dos
    neurônios das camadas ocultas.

    Args:
        model: Modelo de rede neural a ser treinado.
        val_loader: DataLoader de validação.
        loss_function: Função de erro definida pelo usuário.
        device: Dispositivo de aceleração utilizado.
    Output:
        Retorna a média do erro calculado nas iterações da época.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad(): # Desativa o cálculo do gradiente descendente para economizar processamento
        for features, labels in val_loader:
            # --- 1. Enviamos os dados para o dispositivo de aceleração ---
            features = features.to(device)
            labels = labels.to(device)
            # --- 2. Feedforward ---
            # Passamos os features como x1, x2, ..., xn para o modelo gerar uma saída
            predict = model(features)
            # --- 3. Cálculo do erro da saída ---
            out_loss = loss_function(predict, labels)
            # Convertemos o erro (tensor) em um valor numérico
            total_loss += out_loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def testModel(model: nn.Module,
              test_loader: DataLoader,
              mean: pd.Series,
              std: pd.Series,
              device=DEVICE) -> float:
    """
    Função de teste do modelo. Para ajuste das saídas, convertemos valores acima de 0.5 para 1.

    Args:
        model: Modelo de rede neural a ser treinado.
        test_loader: DataLoader de teste.
        mean: pd.Series que contém a média de cada coluna do dataset.
        std: pd.Series que contém o desvio padrão de cada coluna do dataset.
        device: Dispositivo de aceleração utilizado.
    Output:
        Retorna a acurácia média do modelo como float.
    """
    model.eval()
    mse_loss = 0.0
    mean_target = mean.iloc[0]
    std_target  = std.iloc[0]

    with torch.no_grad():
        for features, labels in test_loader:
            # --- 1. Enviamos os dados para o dispositivo de aceleração ---
            features = features.to(device)
            labels = labels.to(device)
            # --- 2. Feedforward ---
            # Passamos os features como x1, x2, ..., xn para o modelo gerar uma saída
            outputs = model(features)

            # Desnormalizar
            outputs_real = outputs * std_target + mean_target
            labels_real  = labels  * std_target + mean_target

            # --- 3. Contagem de samples e dos resultados corretos ---
            mse_loss += torch.mean((outputs_real - labels_real)**2).item()
    
    # --- 5. Calcula a acurácia final ---
    return (mse_loss / len(test_loader))**0.5

def generateSeries(model: nn.Module,
                   test_loader: DataLoader,
                   mean: pd.Series,
                   std: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Função auxiliar que gera a predição do modelo e prepara os dados para serem
    plotados em um gráfico de série temporal.

    Args:
        model: Modelo de rede neural.
        test_loader: DataLoader de teste.
        mean: pd.Series que contém a média de cada coluna do dataset.
        std: pd.Series que contém o desvio padrão de cada coluna do dataset.
    Output:
        Retorna uma tupla contendo umas série com os valores reais e os valores
        preditos, respectivamente.
    """
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            output = model(X_batch)

            preds.append(output.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

        preds = np.concatenate(preds).flatten()
        trues = np.concatenate(trues).flatten()

    true_denorm = trues * std.iloc[0] + mean.iloc[0]
    pred_denorm = preds * std.iloc[0] + mean.iloc[0]

    return true_denorm, pred_denorm

def main():
    # --- 1. Preparação dos dataloaders ---
    train_dl, val_dl, test_dl, mean, std = prepareData(PATH)
    # Exemplo de verificação de um lote de treino
    # features, labels = next(iter(val_dl))
    # print(f"Shape das features do lote de treino: {features.shape}")
    # print(f"Shape dos labels do lote de treino: {labels.shape}")
    # print(f"Exemplo de uma feature normalizada: {features[0]}")

    # --- 2. Instanciação do modelo, da função de erro e do algoritmo de cálculo de gradiente ---
    model = LSTM(hidden_size=HIDDEN_SIZE).to(DEVICE)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(
       model.parameters(),
       lr = ETA 
    )

    # --- 3. Etapa de treinamento e validação ---
    # Armazenamos os erros seguindo o padrão {época: erro obtido}
    train_loss = {}
    val_loss = {}
    for i in range(EPOCHS):
        loss = trainEpoch(model, train_dl, optimizer, loss_func)
        train_loss[i] = loss

        loss = evaluateEpoch(model, val_dl, loss_func)
        val_loss[i] = loss

    plotLosses(train_loss, val_loss, "Erro de treino", "Erro de validação", "Gráfico_de_erros.png")
    true_values, predicted_values = generateSeries(model, test_dl, mean, std)
    plotSeries(true_values, predicted_values, "Average Players", "Predicted", "Grafico_serie_temporal.png")
    accuracy = testModel(model, test_dl, mean, std)
    print("========= Erro de teste =========")
    print(f"{accuracy:.2f}")

if __name__ == "__main__":
    main()
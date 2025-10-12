import os
import torch

from data import prepareData
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose
from view_loss import plotLosses

class LinearModel(nn.Module):
    def __init__(self, in_features=8, out_features=1):
        """
        in_features: Número de features que o dataset possui
        out_features: Classificação binária
        """
        super(LinearModel, self).__init__() # instancia o nn.Module
        self.l1 = nn.Linear(in_features, out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Passa os dados para a camada linear
        # v = Σ(wx) + b
        x = self.l1(x)
        # 2. Função do campo local induzido (não-linear)
        # y = φ(v)
        x = self.sigmoid(x)
        
        return x


def trainModel(model: LinearModel, train_loader: DataLoader, loss_function, optimizer, device="cuda") -> float:
    """
    Função de treinamento do modelo, percorrendo uma época.

    ENTRADA:
        model: Modelo de rede neural a ser treinado.
        train_loader: DataLoader de treinamento.
        loss_function: Função de erro definida pelo usuário.
        optimizer: Método de cálculo da retropropagação.
        device: Dispositivo de aceleração utilizado.
    SAÍDA:
        avg_loss: Média do erro calculado nas iterações da época.
    """
    model.train() # Coloca o modelo em modo de treino
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

def validateModel(model: LinearModel, val_loader: DataLoader, loss_function, device="cuda") -> float:
    """
    Função de validação do modelo, percorrendo uma época. Como não estamos treinando o modelo, não precisamos
    calcular o gradiente descendente estocástico, o qual é necessário para atualizar os pesos sinápticos dos
    neurônios das camadas ocultas.

    ENTRADA:
        model: Modelo de rede neural a ser treinado.
        val_loader: DataLoader de validação.
        loss_function: Função de erro definida pelo usuário.
        device: Dispositivo de aceleração utilizado.
    SAÍDA:
        avg_loss: Média do erro calculado nas iterações da época.
    """
    model.eval() # Coloca o modelo em modo de validação
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

EPOCHS = 20
PATH = os.path.join("db/","csgo_processed.csv")
PROPORTION = (0.8, 0.1, 0.1)

def main():
    # Carregamos os dados localmente
    # Args: (Caminho do dataset, Tupla de proporção=(Treino, Validação, Teste))
    train_loader, val_loader, test_loader = prepareData(PATH, PROPORTION)

    # Definimos o uso de CUDA para o treinamento do modelo
    device = "cuda" if torch.accelerator.is_available() else "cpu"
    # --- 1. Instanciação do modelo ---
    torch.manual_seed(42)
    model = LinearModel().to(device)
    # --- 2. Definição da função de erro ---
    # Soma dos erros quadráticos, como descrito na documentação
    loss_function = nn.MSELoss() # mean squared error
    # --- 3. Definição do gradiente estocástico (otimizador) ---
    learning_rate = 0.01 # eta
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Exemplo de verificação de um lote de treino
    # features, labels = next(iter(train_loader))
    # print(f"Shape das features do lote de treino: {features.shape}")
    # print(f"Shape dos labels do lote de treino: {labels.shape}")
    # print(f"Exemplo de uma feature normalizada: {features[0]}")

    # --- 4. Treinamento e validação do modelo ---
    # Armazenamos os erros seguindo o padrão {época: erro obtido}
    train_loss = {}
    val_loss = {}
    for i in range(EPOCHS):
        loss = trainModel(model, train_loader, loss_function, optimizer)
        train_loss[i] = loss

        loss = validateModel(model, val_loader, loss_function)
        val_loss[i] = loss

    plotLosses(train_loss, val_loss, "Erro de treino", "Erro de validação")

if __name__ == "__main__":
    main()
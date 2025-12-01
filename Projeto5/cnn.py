import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data import prepareData
from torch.utils.data import DataLoader
from view_loss import plotLosses, plotCMatrix

class SimpleCNN(nn.Module):
    def __init__(self, lista_filtros, num_classes):
        super(SimpleCNN, self).__init__()
        layers = []
        in_channels = 3
        
        for num_filtros in lista_filtros:
            # num_filtros: quantidade de neurônios no canal 
            # kernel: janela deslizante
            # padding: estouro do mapa 2D em uma unidade para fora
            layers.append(nn.Conv2d(in_channels, num_filtros, kernel_size=3, padding=1))
            # Batch Norm ajuda a rede a não "viciar" em uma classe
            layers.append(nn.BatchNorm2d(num_filtros))
            layers.append(nn.ReLU()) # escolhido segundo o modelo da AlexNet
            # stride: deslocamento da janela
            layers.append(nn.MaxPool2d(kernel_size=2, stride=1)) # aplica pooling de 2x2 (assim como nos slides da aula 18)
            in_channels = num_filtros # atualiza o tamanho da próxima entrada

        self.feature_extractor = nn.Sequential(*layers) # sequência de camadas convolucionais
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # global pooling  
        self.classifier = nn.Sequential( # classificador mais robusto
            nn.Flatten(),
            # Camada Oculta Densa: Aumentamos a capacidade de raciocínio
            nn.Linear(in_channels, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),             
            # Camada de Saída
            nn.Linear(512, num_classes)   
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
def trainModel(
        model: nn.Module,
        train_loader: DataLoader,
        loss_function: callable,
        optimizer,
        device: str
):
    """
    Função de treinamento do modelo, percorrendo uma época.

    Args:
        model: Modelo de rede neural a ser treinado.
        train_loader: DataLoader de treinamento.
        loss_function: Função de erro definida pelo usuário.
        optimizer: Método de cálculo da retropropagação.
        device: Dispositivo de aceleração utilizado.
    Output:
        avg_loss: Média do erro calculado nas iterações da época.
    """
    model.train()
    total_loss = 0.0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device) # Labels já são int (long) do Dataset

        optimizer.zero_grad()
        
        # Feedforward
        outputs = model(features) 
        
        # Cálculo do Erro (CrossEntropyLoss aceita Logits e Indices de Classe)
        loss = loss_function(outputs, labels)
        
        # Backprop
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validateModel(
        model: nn.Module, 
        val_loader: DataLoader, 
        loss_function: callable, 
        device: str
):
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
        avg_loss: Média do erro calculado nas iterações da época.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(val_loader)

def testModel(
        model: nn.Module, 
        test_loader: DataLoader, 
        device: str
):
    """
    Função de teste do modelo.

    Args:
        model: Modelo de rede neural a ser treinado.
        test_loader: DataLoader de testew.
        device: Dispositivo de aceleração utilizado.
    Output:
        accuracy: Acurácia média do modelo.
        real_lst: Lista contendo os labels de cada amostra.
        predicted_lst: Lista contendo o label predito pelo modelo por amostra.
    """
    model.eval()
    correct_cases = 0
    total_samples = 0

    real_lst = []
    predicted_lst = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)

            _, predicted = torch.max(outputs.data, 1)
            
            total_samples += labels.size(0)
            correct_cases += (predicted == labels).sum().item()

            real_lst.extend(labels.cpu().numpy())
            predicted_lst.extend(predicted.cpu().numpy())
    
    return 100 * correct_cases / total_samples, real_lst, predicted_lst

def visualize_batch(dataloader, classes):
    # Pega um lote de imagens
    images, labels = next(iter(dataloader))
    
    plt.figure(figsize=(12, 4))
    for idx in range(5):
        ax = plt.subplot(1, 5, idx+1)
        # O PyTorch é (C, H, W), o Matplotlib quer (H, W, C)
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.title(classes[labels[idx]])
        plt.axis("off")
    plt.show()

EPOCHS = 40
ETA = 1e-4
INPUT_SIZE = 3 * 64 * 64
FILTERS = [16, 32, 64]
OUTPUT_SIZE = 16
#L2 = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = [
    "Bug", "Dark", "Dragon", "Electric",
    "Fighting", "Fire", "Ghost", "Grass",
    "Ground", "Ice", "Normal", "Poison", 
    "Psychic", "Rock", "Steel", "Water"
]

def main():
    # Carrega dados do Pokémon
    train_loader, val_loader, test_loader = prepareData("db/pokemon.csv", "db/images/")
    #visualize_batch(train_loader, LABELS)
    
    # --- 1. Instancia CNN ---
    model = SimpleCNN(lista_filtros=FILTERS, num_classes=OUTPUT_SIZE).to(DEVICE)

    # --- 2. Função de Erro correta para Multiclasse ---
    loss_function = nn.CrossEntropyLoss() 

    # --- 3. Otimizador ---
    optimizer = optim.Adam(model.parameters(), lr=ETA)#, weight_decay=L2)

    train_loss = {}
    val_loss = {}
    best_val_loss = float('inf')

    for i in range(EPOCHS):
        t_loss = trainModel(model, train_loader, loss_function, optimizer, DEVICE)
        v_loss = validateModel(model, val_loader, loss_function, DEVICE)
        
        train_loss[i] = t_loss
        val_loss[i] = v_loss
        # Lógica de memorizar o melhor modelo
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            # Salva os pesos no disco
            torch.save(model.state_dict(), "best_cnn_model.pth")
            print(f"Epoch {i+1}: Melhor modelo salvo! (Val Loss: {v_loss:.4f})")

        if (i+1) % 5 == 0:
            print(f"Epoch {i+1}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
    # Plotagem dos erros
    plotLosses(train_loss, val_loss, "Erro de treino", "Erro de validação")

    # --- 4. Avaliação Final ---
    model.load_state_dict(torch.load("best_cnn_model.pth"))
    acc, labels, predicted = testModel(model, test_loader, DEVICE)
    print(f"\nAcurácia final no teste: {acc:.2f}%")

    plotCMatrix(labels, predicted, LABELS, "assets/CNN_ConfusionMatrix.png")

if __name__ == "__main__":
    main()
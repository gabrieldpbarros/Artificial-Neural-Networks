import torch
import torch.nn as nn
import torch.optim as optim

from data import prepareData
from torch.utils.data import DataLoader
from view_loss import plotLosses, plotCMatrix

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Args:
            input_size (int): Total de pixels da imagem (C * H * W). Ex: 3*64*64 = 12288.
            hidden_sizes (list): Lista com o número de neurônios em cada camada oculta.
                                 Ex: [512, 256] cria duas camadas ocultas.
            output_size (int): Número de classes para classificação (Ex: 16 tipos de Pokémon).
        """
        super(MLP, self).__init__()
        # Lista para armazenar as camadas sequencialmente
        layers = []
        last_size = input_size

        # --- Construção Dinâmica das Camadas Ocultas ---
        for hidden_dim in hidden_sizes:
            # --- 1. Camada Linear (v = Σ(wx) + b) ---
            layers.append(nn.Linear(last_size, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            # --- 2. Função de Ativação Não-Linear (v = max(0,x)) ---
            layers.append(nn.ReLU())
            # --- 3. Dropout (Opcional, mas recomendado para evitar overfitting) ---
            layers.append(nn.Dropout(0.5))
            # Atualiza o tamanho de entrada da próxima camada
            last_size = hidden_dim

        # --- Camada de Saída ---
        # Conecta a última camada oculta ao número de classes (Tipos de Pokémon)
        layers.append(nn.Linear(last_size, output_size))
        
        # Empacota tudo em um container Sequencial
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # --- 1. Flattening (Achatar a imagem) ---
        # Transforma o Tensor de imagem (Batch, 3, 64, 64) -> Vetor (Batch, 12288)
        # Sem isso, a camada Linear não aceita a imagem
        x = x.view(x.size(0), -1) 
        
        # --- 2. Passagem pela rede (Linear -> ReLU -> Dropout -> ... -> Saída) ---
        x = self.network(x)
        
        # Não usamos self.sigmoid() ou softmax() aqui no final.
        # O PyTorch (nn.CrossEntropyLoss) prefere receber os "Logits" (valores brutos).
        # Ele aplicará o Softmax internamente de forma mais estável.
        
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

EPOCHS = 30
ETA = 0.01
INPUT_SIZE = 3 * 64 * 64
OUTPUT_SIZE = 16
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

    # --- 1. Instancia MLP ---
    # Exemplo MLP:
    input_pixels = 3 * 64 * 64
    model = MLP(input_size=INPUT_SIZE, hidden_sizes=[512, 256], output_size=16).to(DEVICE)
    
    # Exemplo CNN (Descomente para usar):
    # model = SimpleCNN(lista_filtros=[32, 64], num_classes=16).to(DEVICE)

    # --- 2. Função de Erro correta para Multiclasse ---
    loss_function = nn.CrossEntropyLoss() 

    # --- 3. Otimizador ---
    optimizer = optim.SGD(model.parameters(), lr=ETA, momentum=0.9)

    train_loss = {}
    val_loss = {}

    for i in range(EPOCHS):
        t_loss = trainModel(model, train_loader, loss_function, optimizer, DEVICE)
        v_loss = validateModel(model, val_loader, loss_function, DEVICE)
        
        train_loss[i] = t_loss
        val_loss[i] = v_loss
        
        if (i+1) % 5 == 0:
            print(f"Epoch {i+1}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
    # Plotagem dos erros
    plotLosses(train_loss, val_loss, "Erro de treino", "Erro de validação")

    # --- 4. Avaliação Final ---
    acc, labels, predicted = testModel(model, test_loader, DEVICE)
    print(f"\nAcurácia final no teste: {acc:.2f}%")

    plotCMatrix(labels, predicted, LABELS, "assets/MLP_ConfusionMatrix.png")

if __name__ == "__main__":
    main()
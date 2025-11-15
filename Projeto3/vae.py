import os
import numpy as np
import torch

from data import loadData
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose
from view_latent_space import visualize_latent_space_pca, visualize_latent_space_tsne

class VAEmodel(nn.Module):
    """ Modelo extraído de um exemplo do Kaggle, com adaptações """
    def __init__(self, latent_dims, hidden_dims, image_shape):
        super(VAEmodel, self).__init__()
        
        self.latent_dims = latent_dims # Size of the latent space layer
        self.hidden_dims = hidden_dims # List of hidden layers number of filters/channels
        self.image_shape = image_shape # Input image shape
        
        self.last_channels = self.hidden_dims[-1]
        self.in_channels = self.image_shape[0]
        # Simple formula to get the number of neurons after the last convolution layer is flattened
        self.flattened_channels = int(self.last_channels*(self.image_shape[1]/(2**len(self.hidden_dims)))**2) 
       
        # --- 1. Codificador (Encoder) ---
        # For each hidden layer we will create a Convolution Block
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels,
                              out_channels=h_dim,
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            
            self.in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # --- 2. Espaço Latente ---
        # Here are our layers for our latent space distribution
        self.fc_mu = nn.Linear(self.flattened_channels, latent_dims)
        self.fc_var = nn.Linear(self.flattened_channels, latent_dims)
        
        # Decoder input layer
        self.decoder_input = nn.Linear(latent_dims, self.flattened_channels)
        
        # --- 3. Decodificador (Decoder) ---
        # For each Convolution Block created on the Encoder we will do a symmetric Decoder with the same Blocks, but using ConvTranspose
        self.hidden_dims.reverse()
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.in_channels,
                                       out_channels=h_dim,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            
            self.in_channels = h_dim
        
        self.decoder = nn.Sequential(*modules)
        
        # --- 4. Camada de saída ---
        # The final layer the reconstructed image have the same dimensions as the input image
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.image_shape[0],
                      kernel_size=3,
                      padding=1),
            nn.Sigmoid()
        )
        
    def get_latent_dims(self):
        
        return self.latent_dims
        
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes. (z = e_theta(x))

        Entrada -> codificador -> espaço latente
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return [mu, log_var]
    
    def decode(self, z):
        """
        Maps the given latent codes onto the image space. (x' = d_phi(z))

        Espaço latente -> decodificador -> camada de saída
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.last_channels, int(self.image_shape[1]/(2**len(self.hidden_dims))), int(self.image_shape[1]/(2**len(self.hidden_dims))))
        result = self.decoder(result)
        result = self.final_layer(result)
        
        return result
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def forward(self, input):
        """
        Forward method which will encode and decode our image.
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        
        return  [self.decode(z), input, mu, log_var, z]
    
    def loss_function(self, recons, input, mu, log_var):
        """
        Computes VAE loss function
        """
        recons_loss = nn.functional.binary_cross_entropy(recons.reshape(recons.shape[0],-1),
                                                         input.reshape(input.shape[0],-1),
                                                         reduction="none").sum(dim=-1)
       
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        
        loss = (recons_loss + kld_loss).mean(dim=0)
        
        return loss
        
    def sample(self, num_samples, device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        """
        z = torch.randn(num_samples, self.latent_dims)
        z = z.to(device)
        samples = self.decode(z)
        
        return samples
    
    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        """
        return self.forward(x)[0]
    
    def interpolate(self, starting_inputs, ending_inputs, device, granularity=10):
        """
        This function performs a linear interpolation in the latent space of the autoencoder
        from starting inputs to ending inputs. It returns the interpolation trajectories.
        """
        mu, log_var = self.encode(starting_inputs.to(device))
        starting_z = self.reparameterize(mu, log_var)
        
        mu, log_var = self.encode(ending_inputs.to(device))
        ending_z  = self.reparameterize(mu, log_var)
        
        t = torch.linspace(0, 1, granularity).to(device)
        
        intep_line = (
            
            torch.kron(starting_z.reshape(starting_z.shape[0], -1), (1 - t).unsqueeze(-1))+
            torch.kron(ending_z.reshape(ending_z.shape[0], -1), t.unsqueeze(-1))
            
        )
    
        decoded_line = self.decode(intep_line).reshape(
            (
                starting_inputs.shape[0],
                t.shape[0]
            )
            + (starting_inputs.shape[1:])
        )
        return decoded_line

DEVICE = "cuda"
LATENT_DIMS = 128
HIDDEN_DIMS = [32, 64, 128, 256] # Aumentando a profundidade progressivamente
IMAGE_SHAPE = (3, 64, 64) # (Canais, Altura, Largura)
BATCH_SIZE = 32
LR = 0.001 
EPOCHS = 50

def trainEpoch(model: nn.Module, loader: DataLoader, optimizer, device=DEVICE) -> float:
    """
    Função de treinamento do modelo, percorrendo uma época.

    Args:
        model: Modelo de rede neural a ser treinado.
        loader: DataLoader de treinamento.
        optimizer: Método de cálculo da retropropagação.
        device: Dispositivo de aceleração utilizado.
    Output:
        avg_loss: Média do erro calculado nas iterações da época.
    """
    model.train()
    total_loss = 0.0

    for idx, (data, _) in enumerate(loader): # a iteração é um pouco diferente do MLP, pois não estamos usando labels
        # --- 1. Envio para o dispositivo de aceleração e definição do gradiente para zero ---
        data = data.to(device)
        optimizer.zero_grad()

        # --- 2. Feedforward ---
        results = model(data)
        # Seguimos o retorno do método forward do modelo
        recons = results[0]
        curr_input = results[1]
        mu = results[2]
        log_var = results[3]

        # --- 3. Cálculo do erro ---
        loss = model.loss_function(recons, curr_input, mu, log_var)

        # --- 4. Feedback e atualização dos pesos ---
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validateEpoch(model: nn.Module, loader: DataLoader, device=DEVICE) -> float:
    """
    Função de validação do modelo, percorrendo uma época. Como não estamos treinando o modelo, não precisamos
    calcular o gradiente descendente estocástico, o qual é necessário para atualizar os pesos sinápticos dos
    neurônios das camadas ocultas.

    Args:
        model: Modelo de rede neural a ser treinado.
        loader: DataLoader de validação.
        device: Dispositivo de aceleração utilizado.
    Output:
        avg_loss: Média do erro calculado nas iterações da época.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for idx, (data, _) in enumerate(loader):
            # --- 1. Envio para o dispositivo de aceleração ---
            data = data.to(device)

            # --- 2. Feedforward ---
            results = model(data)
            recons = results[0]
            curr_input = results[1]
            mu = results[2]
            log_var = results[3]

            # --- 3. Cálculo do erro ---
            loss = model.loss_function(recons, curr_input, mu, log_var)
            total_loss += loss.item()

    return total_loss / len(loader)

def testModel(model: nn.Module, loader: DataLoader, device=DEVICE):
    """
    Executa o modelo no conjunto de teste e extrai os vetores latentes (mu)
    e os rótulos para visualização posterior.

    Args:
        model: Modelo autoencoder variacional.
        loader: DataLoader de validação.
        device: Dispositivo de aceleração utilizado.
    Output:
        avg_loss: Média do erro calculado nas iterações da época.
    """
    model.eval()
    latent_vectors = []
    labels_list = []
    
    with torch.no_grad():
        for data, labels in loader:
            # --- 1. Envio para o dispositivo de aceleração ---
            data = data.to(device)
            
            # --- 2. Cálculo do vetor latente ---
            # Queremos apenas o 'mu' para visualização (posição central do cluster)
            mu, _ = model.encode(data)
            
            # --- 3. Retorno para CPU e conversão para numpy para usar no matplotlib/sklearn ---
            latent_vectors.append(mu.cpu().numpy())
            labels_list.append(labels.numpy())
    
    # --- 4. Concatenação de todos os batches em dois grandes arrays ---
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    
    return latent_vectors, labels_list

def main():
    lbl_path = "./db/pokemons_labels.csv"
    img_path = "./db/pokemon"
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SHAPE[1], IMAGE_SHAPE[2])), # Garante que todas tenham o mesmo tamanho
        transforms.ToTensor(), # Converte imagem [0-255] para Tensor [0-1]
    ])

    # --- 1. Carregamento dos dados ---
    train_loader, val_loader, test_loader = loadData(lbl_path, img_path, transform, BATCH_SIZE)\
    
    # --- 2. Instanciação do modelo ---
    vae = VAEmodel(LATENT_DIMS, HIDDEN_DIMS, IMAGE_SHAPE).to(DEVICE)

    # --- 3. Definição do otimizador ---
    optimizer = torch.optim.Adam(vae.parameters(), lr=LR)

    # --- 4. Loop de treinamento e validação ---
    best_val_loss = float('inf')
    
    print("--- FASE DE TREINAMENTO ----")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # --- 1. Treino ---
        train_loss = trainEpoch(vae, train_loader, optimizer, DEVICE)
        
        # --- 2. Validação ---
        val_loss = validateEpoch(vae, val_loader, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Salvar o melhor modelo baseado na validação
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(vae.state_dict(), 'best_vae_model.pth')
            print("-> Modelo salvo (Melhor loss de validação)")
        
    # --- 5. Fase de teste do modelo ---
    print("--- FASE DE TESTE ----")
    try:
        class_names = train_loader.dataset.dataset.classes 
    except AttributeError:
        class_names = None # Caso não consiga recuperar os nomes automaticamente
    z_test, labels_test = testModel(vae, test_loader, DEVICE)
    visualize_latent_space_pca(z_test, labels_test, class_names)
    visualize_latent_space_tsne(z_test, labels_test, class_names)

if __name__ == "__main__":
    main()
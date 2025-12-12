import os
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, Tuple

class CustomDB(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            labels_dict: Dict[int, str],
            img_dir: str,
            transform: callable = None
    ):
        """
        Args:
            df: DataFrame contendo nomes e tipos.
            img_dir: Caminho para a pasta das imagens.
            label_map: Dicionário mapeando {'Tipo': Inteiro}. Garante consistência.
            transform: Transformações do PyTorch.
        """
        # --- Dados do DataFrame ---
        self.full_df = df.reset_index(drop=True) # Garante que os índices sejam 0, 1, 2...
        self.labels_dic = labels_dict # dicionário que associa os tipos a um valor numérico
        # --- Diretório das imagens ---
        self.img_dir = img_dir
        # --- Transformação ---
        self.transform = transform

    def __len__(self):
        return len(self.full_df.index)
    
    def __getitem__(self, index):
        """
        Método que busca um índice no dataset e retorna a imagem acompanhada do seu
        label, em ordem.
        """
        row = self.full_df.iloc[index]
        pk_name = row["Name"] # nome do pokemon
        pk_type = row["Type1"] # tipo do pokemon

        # --- 1. Encontramos o caminho para a imagem relativa ao pokemon do índice ---
        img_path = os.path.join(self.img_dir, f"{pk_name}.png")

        # --- 2. Carregamento e tratamento da imagem ---
        try:
            image = Image.open(img_path)
            # Como as imagens do dataset são no formato .png, sem fundo, adicionamos um fundo.
            if image.mode == "RGBA": # fundo transparente
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3]) # 3 é o canal alpha
                image = background
            else:
                image = image.convert("RGB")
        except FileNotFoundError:
            print(f"Aviso: Imagem não encontrada {img_path}.")

        # --- 3. Extraímos o tipo do pokemon ---
        numerical_type = self.labels_dic[pk_type]

        # --- 4. Aplica a transformação na imagem ---
        if self.transform:
            image = self.transform(image)

        return image, int(numerical_type)
    
def prepareData(
        labels_path: str,
        images_path: str,
        batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Carrega os dados em formato de um DataLoader.

    Args:
        labels_path: Caminho para o arquivo .csv que contém os rótulos.
        images_path: Diretório que armazena as imagens.
        batch_size: Tamanho dos lotes de treinamento.

    Output:
        Dados separados conforme a proporção definida internamente em conjuntos de treino,
        validação e teste
    """
    proportions = (0.7, 0.3) # proporções de divisão dos conjuntos (treino, (validação e teste))
    # --- 1. Carregamento do DataFrame e definição de elementos essenciais ---
    full_df = pd.read_csv(labels_path)
    df = filterData(full_df)
    # Dicionário de tipos de pokemons
    labels_df = sorted(df["Type1"].unique())
    #print(labels_df)
    types_dict = {pk_type: i for i, pk_type in enumerate(labels_df)}

    # --- 2. Separação dos conjuntos de treino, validação e teste ---
    # Conjunto de treino
    train_df, temp = train_test_split(
        df,
        test_size=proportions[1],
        stratify=df["Type1"],
        random_state=42
    )
    # Conjuntos de validação e teste
    val_df, test_df = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp["Type1"],
        random_state=42
    )

    # --- 3. Definição das transformações
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomHorizontalFlip(p=0.5), # Espelha a imagem aleatoriamente
        #transforms.RandomRotation(15),
        #transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=255), # Rotaciona a imagem aleatoriamente
        transforms.ToTensor(),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # --- 4. Instanciação dos datasets e criação dos dataloaders ---
    train_ds = CustomDB(train_df, types_dict, images_path, train_transform)
    val_ds = CustomDB(val_df, types_dict, images_path, val_test_transform)
    test_ds = CustomDB(test_df, types_dict, images_path, val_test_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 # colocar em 2 no Colab, caso identifique algum erro
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader

def filterData(
        full_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove dados que estejam causando problemas para a peparação dos dataloaders.
    Como há um tipo de pokemon com apenas 3 exemplos, vamos removê-lo para permitir
    que a função train_test_split seja executada corretamente, além de outros tipos
    que possam estar dificultando esse processo.

    Args:
        full_df: DataFrame original, que será alterado.
    Output:
        Retornamos o DataFrame filtrado dos dados problemáticos.
    """
    # --- 1. Conta quantas imagens existem por tipo ---
    counts = full_df['Type1'].value_counts()

    # --- 2. Define um limite mínimo (Ex: 20 imagens) ----
    MIN_SAMPLES = 20
    valid_types = counts[counts >= MIN_SAMPLES].index
    #print(valid_types)

    # --- 3. Filtra o DataFrame mantendo apenas os tipos válidos ---
    return full_df[full_df['Type1'].isin(valid_types)].copy()

if __name__ == "__main__":
    x, y, z = prepareData("db/pokemon.csv", "db/images/")
    # Exemplo de verificação de um lote de treino
    features, labels = next(iter(x))
    print(f"Shape das features do lote de treino: {features.shape}")
    print(f"Shape dos labels do lote de treino: {labels.shape}")
    print(f"Exemplo de uma feature normalizada: {features[0]}")
import os
import pandas as pd
import pokebase as pb

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from typing import Tuple

EXCEPTIONS = (
    "201-f.png", "386-attack.png", "386-defense.png", "386-normal.png",
    "386-speed.png", "412-plant.png", "412-sandy.png", "412-trash.png",
    "413-plant.png", "413-trash.png", "421-overcast.png", "421-sunshine.png",
    "422-east.png", "422-west.png", "423-east.png", "423-west.png",
    "487-altered.png", "487-origin.png", "492-land.png", "492-sky.png",
    "493-normal.png", "521f.png", "550-blue-striped.png", "550-red-striped.png",
    "555-standard.png", "585-autumn.png", "585-spring.png", "585-summer.png",
    "585-winter.png", "586-autumn.png", "586-spring.png", "586-summer.png",
    "586-winter.png", "592f.png", "593f.png", "641-incarnate.png",
    "641-therian.png", "642-incarnate.png", "642-therian.png", "645-incarnate.png",
    "645-therian.png", "647-ordinary.png", "647-resolute.png", "648-pirouette.png",
    "666-elegant.png", "668f.png", "676-diamond.png", "676-heart.png",
    "676-star.png", "678f.png", "681-blade.png", "681-shield.png"
)

class CustomDB(Dataset):
    def __init__(self, labels_path: str, img_dir: str, transform: callable=None):
        """
        Args:
            labels_path: Caminho para os rótulos que geramos na função createPokemonLabels.
            img_dir: Caminho para o diretório com as imagens dos pokémons.
            transform: Transformações que fazemos nas imagens para facilitar o treinamento.
        """
        self.data = pd.read_csv(labels_path)
        self.img_dir = img_dir
        self.transform = transform

        # Esta etapa converte os tipos dos pokémons para valores numéricos, permitindo que
        # o modelo possa compreender os rótulos.
        self.classes = sorted(self.data["Type1"].unique())
        # Dicionário que usaremos para retornar do valor numérico para o tipo do pokémon
        self.classes_dic = {cls_name: i for i, cls_name in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # --- 1. Selecionamos a imagem correspondente ---
        row = self.data.iloc[index]
        img = str(row["Filename"])
        img_path = os.path.join(self.img_dir, img)

        # --- 2. Carregamento e tratamento da imagem ---
        # Como as imagens do dataset são no formato .png, sem fundo, adicionamos um fundo.
        try:
            image = Image.open(img_path)

            if image.mode == "RGBA": # fundo transparente
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3]) # 3 é o canal alpha
                image = background
            else:
                image = image.convert("RGB")
        except FileNotFoundError:
            print(f"Aviso: Imagem não encontrada {img_path}.")
        
        # --- 3. Carrega o label e converte para um valor numérico ---
        lbl_str = row["Type1"]
        label = self.classes_dic[lbl_str]

        # --- 4. Aplica a transformação na imagem ---
        if self.transform:
            image = self.transform(image)

        return image, label

def createPokemonLabels(img_dir: str, output_dir: str) -> None:
    """
    Cria um arquivo .csv contendo os nomes e tipos de cada pokémon. Como o dataset
    do Kaggle não possui nenhuma das duas informações, apenas o id da pokédex,
    precisamos definí-las para poder desenvolver o VAE. Além disso, alguns dos
    pokémons possuem no label da imagem certas especificações, como megaevoluções.
    Como a API do pokebase espera receber apenas o ID do pokémon, fazemos um
    tratamento desses casos especiais, ou seja, uma consulta especial. Como estamos
    fazendo inúmeras consultas à API, é de se esperar que essa função demore.

    Args:
        img_dir: Diretório do dataset de imagens.
        output_dir: Diretório de destino do .csv com os rótulos.
    """
    data = []
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

    for file in tqdm(files): # tqdm é apenas um detalhe de estética
        # try-catch para facilitar a correção de bugs, caso ocorra
        if file in EXCEPTIONS:
            continue

        try:
            # Remove a extensão do arquivo
            clean_name = os.path.splitext(file)[0] # ex: "6-mega-x" ou "25"
            pokemon_obj = None

            # LÓGICA PARA TRATAR NOMES COM HÍFEN (Megas, Alolan, etc)
            if '-' in clean_name:
                parts = clean_name.split('-', 1) # Divide apenas no primeiro hífen
                base_id = parts[0]    # "6"
                suffix = parts[1]     # "mega-x"
            
                try:
                    # --- 1. Descobrir o nome do Pokémon base pelo ID ---
                    base_name = pb.pokemon(int(base_id)).name
                
                    # --- 2. Montar o nome que a API espera (charizard-mega-x) ---
                    api_full_name = f"{base_name}-{suffix}"
                
                    # --- 3. Buscar os dados da forma específica ---
                    pokemon_obj = pb.pokemon(api_full_name)
                
                except:
                    # Fallback: Se falhar (ex: nome da imagem não bate com a API), 
                    # tenta pegar só o tipo do Pokémon base para não perder a imagem
                    pokemon_obj = pb.pokemon(int(base_id))
                    print(f"Aviso: Usando tipo base para {clean_name}")

            else:
                # Caso padrão: apenas número
                pokemon_obj = pb.pokemon(int(clean_name))

            # Extrair o Tipo Primário
            if pokemon_obj:
                primary_type = pokemon_obj.types[0].type.name
            
                data.append({
                    'Filename': file,
                    'Name': pokemon_obj.name,
                    'Type1': primary_type
                })
            
        except Exception as e:
            print(f"Erro crítico ao processar {file}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_dir, index=False)
    print(df['Type1'].value_counts()) # Mostra quantos de cada tipo encontrou

def loadData(label_path: str,
             img_path: str,
             transformation: callable,
             batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Carrega os dados em formato de um DataLoader customizado.

    Args:
        label_path: Caminho para o arquivo .csv que contém os rótulos.
        img_path: Diretório que armazena as imagens.
        transformation: Transformação que redimenziona a imagem e converte para um tensor.
        batch_size: Tamanho dos lotes de treinamento.

    Output:

    """
    # --- 1. Montagem do dataset ---
    dataset = CustomDB(label_path, img_path, transformation)

    # --- 2. Separação dos conjuntos ---
    full_size = len(dataset)
    train_size = int(0.8 * full_size)
    val_size = int(0.1 * full_size)
    test_size = full_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # --- 3. Criação dos DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 # colocar em 2 no Colab, caso identifique algum erro
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # não precisamos embaralhar
        num_workers=0 # colocar em 2 no Colab, caso identifique algum erro
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 # colocar em 2 no Colab, caso identifique algum erro
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    IMAGE_DIR = "./db/pokemon"
    OUTPUT_DIR = "./db/pokemons_labels.csv"
    createPokemonLabels(IMAGE_DIR, OUTPUT_DIR)
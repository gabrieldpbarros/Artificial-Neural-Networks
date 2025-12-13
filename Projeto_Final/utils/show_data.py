import os
import matplotlib.pyplot as plt

from load_data import getDataLoaders
from torch.utils.data import DataLoader
from typing import Tuple

def sampleBatch(
    dataloader: DataLoader,
    classes: Tuple[str] = ["NORMAL", "PNEUMONIA"]
) -> None:
    """
    Função de visualização do lote que foi extraído na criação dos DataLoaders

    Args:
        dataloader: DataLoader com imagens dos raios-x
        classes: Labels das imagens (normal e pneumonia)
    """
    # Pega um lote de imagens
    images, labels = next(iter(dataloader))
    
    plt.figure(figsize=(12, 4))
    for idx in range(5):
        _ = plt.subplot(1, 5, idx+1)
        # O PyTorch é (C, H, W), o Matplotlib quer (H, W, C)
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.title(classes[labels[idx]])
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_path = os.path.join(project_root, 'db')

    train_dl, _, _ = getDataLoaders(db_path)
    sampleBatch(train_dl)
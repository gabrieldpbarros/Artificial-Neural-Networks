import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_latent_space_pca(latent_vectors, labels, class_names):
    """
    Aplica PCA para reduzir o espaço latente para 2D e plota os clusters.
    
    Args:
        latent_vectors: Array numpy com dimensão (N_amostras, Latent_Dim)
        labels: Array numpy com os índices das classes (0, 1, 2...)
        class_names: Lista ou Dicionário mapeando índice -> Nome (ex: 'Fire', 'Water')
    """
    # --- 1. Aplica PCA para reduzir para 2 componentes (2D) ---
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    # Quanto da variância é explicada
    explained_variance = pca.explained_variance_ratio_
    print(f"Variância explicada por PC1: {explained_variance[0]:.2%}")
    print(f"Variância explicada por PC2: {explained_variance[1]:.2%}")
    print(f"Total: {sum(explained_variance):.2%}")

    # --- 2. Plotagem ---
    plt.figure(figsize=(10, 8))
    
    # Usamos um colormap que suporte muitas classes (tab20 é bom para até 20 tipos)
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                          c=labels, cmap='tab20', alpha=0.7, s=15)
    
    plt.colorbar(scatter, label="Classes (Índices)")
    plt.title(f"Projeção PCA do Espaço Latente (Total Var: {sum(explained_variance):.1%})")
    plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.1%})")
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.1%})")
    
    # Tentar criar uma legenda com nomes reais (pode ficar poluído se forem muitos)
    if class_names:
        # Lógica simplificada para printar quais índices são quais tipos no console
        print("\nLegenda de Classes:")
        for idx, name in enumerate(class_names):
            print(f"{idx}: {name}")

    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_latent_space_tsne(latent_vectors, labels, class_names):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab20', alpha=0.7, s=15)
    plt.colorbar(scatter, label="Classes")
    plt.title("Visualização t-SNE do Espaço Latente")

    if class_names:
        # Lógica simplificada para printar quais índices são quais tipos no console
        print("\nLegenda de Classes:")
        for idx, name in enumerate(class_names):
            print(f"{idx}: {name}")

    plt.grid(True, alpha=0.3)
    plt.show()
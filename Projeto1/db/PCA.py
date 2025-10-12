# -*- coding: utf-8 -*-

#==============================================================================
# SCRIPT COMPLETO PARA IMPLEMENTAÇÃO DE PCA A PARTIR DE UM ARQUIVO CSV
#==============================================================================
# Este script demonstra o fluxo de trabalho completo para aplicar a Análise de
# Componentes Principais (PCA) em um conjunto de dados carregado de um CSV.
#
# Etapas incluídas:
# 1. Carregamento do dataset a partir de um arquivo CSV.
# 2. Separação de features (X) e label (y).
# 3. Padronização dos dados (passo crucial para o PCA).
# 4. Análise da variância para determinar o número ideal de componentes.
# 5. Aplicação do PCA para reduzir a dimensionalidade.
# 6. Treinamento de um modelo de machine learning com os dados transformados.
# 7. Visualização dos dados reduzidos a 2 dimensões.
#==============================================================================

# --- Passo 0: Importar as bibliotecas necessárias ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import sys


#==============================================================================
# ▼▼▼ CONFIGURAÇÃO DO USUÁRIO ▼▼▼
#==============================================================================
# Altere as duas linhas abaixo com as informações do seu dataset.

# 1. Insira o nome do seu arquivo CSV. O arquivo deve estar na mesma pasta.
NOME_DO_ARQUIVO_CSV = 'csgo_processed.csv'  # Exemplo: 'dados_clientes.csv'

# 2. Insira o nome exato da coluna que contém o label (a variável alvo).
NOME_DA_COLUNA_LABEL = 'round_winner_numeric' # Exemplo: 'churn' ou 'target'

#==============================================================================
# ▲▲▲ FIM DA CONFIGURAÇÃO ▲▲▲
#==============================================================================


# --- Passo 1: Carregar e Preparar os Dados ---
print("--- 1. Carregamento e Preparação dos Dados ---")
try:
    df = pd.read_csv(NOME_DO_ARQUIVO_CSV)
    print(f"Arquivo '{NOME_DO_ARQUIVO_CSV}' carregado com sucesso!")
except FileNotFoundError:
    print(f"ERRO: O arquivo '{NOME_DO_ARQUIVO_CSV}' não foi encontrado.")
    print("Por favor, verifique se o nome do arquivo está correto e se ele está na mesma pasta que o script.")
    sys.exit() # Encerra o script se o arquivo não for encontrado

print("\nVisualização das primeiras linhas do dataset:")
print(df.head())
print(f"\nDimensões do dataset completo: {df.shape}")

# Verificando se a coluna de label existe no DataFrame
if NOME_DA_COLUNA_LABEL not in df.columns:
    print(f"ERRO: A coluna de label '{NOME_DA_COLUNA_LABEL}' não foi encontrada no arquivo CSV.")
    print(f"Colunas disponíveis: {list(df.columns)}")
    sys.exit()

# Separando as features (X) do label (y)
X = df.drop(NOME_DA_COLUNA_LABEL, axis=1)
y = df[NOME_DA_COLUNA_LABEL]

# Lidando com possíveis colunas não-numéricas nas features
X = X.select_dtypes(include=np.number)
print(f"\nFeatures numéricas selecionadas para o PCA. Dimensões de X: {X.shape}")
print(f"Dimensões do label (y): {y.shape}\n")


# --- Passo 2: Padronização das Features (Scaling) ---
print("--- 2. Padronização das Features ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Dados padronizados com sucesso.\n")


# --- Passo 3: Análise de Variância e Escolha do Número de Componentes ---
print("--- 3. Análise da Variância Explicada pelo PCA ---")
pca_analysis = PCA()
pca_analysis.fit(X_scaled)
cumulative_variance = np.cumsum(pca_analysis.explained_variance_ratio_)

print("Variância acumulada por número de componentes:")
for i, cum_var in enumerate(cumulative_variance):
    print(f"  - {i+1} componente(s): {cum_var:.2%} da variância total explicada.")

# Plotando o gráfico de variância acumulada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.title('Gráfico de Variância Acumulada Explicada')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Porcentagem da Variância Acumulada')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='-', label='Limite de 95% de variância')
plt.legend(loc='best')
plt.show()


# --- Passo 4: Aplicar o PCA para Redução de Dimensionalidade ---
print("\n--- 4. Aplicando o PCA para Reduzir a Dimensionalidade ---")
pca = PCA(n_components=0.95) # Retém 95% da variância
X_pca = pca.fit_transform(X_scaled)

print(f"Dimensões do dataset original (features): {X_scaled.shape}")
print(f"Dimensões do dataset após PCA: {X_pca.shape}")
print(f"Número de componentes escolhido para reter 95% da variância: {pca.n_components_}\n")


# --- Passo 5: Treinamento de um Modelo de Machine Learning ---
print("--- 5. Treinando um Modelo de Classificação com os Dados Reduzidos ---")
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Acurácia do modelo nos dados de teste: {accuracy:.4f}")
print("Matriz de Confusão:")
print(conf_matrix, "\n")


# --- Passo 6: Visualização dos Componentes Principais ---
print("--- 6. Visualizando os Dados em 2D ---")
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
df_pca_2d = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])
df_pca_2d['label'] = y.values # Adiciona o label original

plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='label', data=df_pca_2d, palette='viridis', s=80, alpha=0.8)
plt.title('Visualização dos Dados em 2D após PCA')
plt.xlabel(f'Primeiro Componente Principal (explica {pca_2d.explained_variance_ratio_[0]:.2%} da variância)')
plt.ylabel(f'Segundo Componente Principal (explica {pca_2d.explained_variance_ratio_[1]:.2%} da variância)')
plt.legend(title='Label')
plt.grid(True)
plt.show()

print("\n--- Script Concluído ---")
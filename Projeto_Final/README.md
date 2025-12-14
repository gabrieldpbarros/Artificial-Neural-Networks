# Desenvolvimento de modelos de redes neurais convolucionais para o diagnóstico de casos de pneumonia através de imagens de radiografias torácicas.

**Discente**: Gabriel Delgado Panovich de Barros - 176313

## Resumo

A implementação de algoritmos para o apoio à decisão clínica, como diagnóstico por imagem, é um desafio fundamentado na complexidade de definição de métodos de confiabilidade para as saídas fornecidas por esses programas. O uso dessas ferramentas é essencial para automatizar processos onerosos, agilizando o acesso do paciente a tratamentos essenciais e, situacionalmente, urgentes. Este projeto visa fornecer uma abordagem para esse gargalo através de arquiteturas de Redes Neurais Convolucionais (CNNs), aplicadas à detecção de pneumonia em radiografias torácicas. Ao comparar modelos próprios com modelos de referência (estado da arte), utilizando técnicas de *Data Augmentation* para potencializar a generalização do modelo, busca-se atingir uma arquitetura que equilibre custo e precisão, de modo que as aplicações do modelo sejam viáveis no diagnóstico por imagem. **(adicionar resultados)**

## Sobre o programa

### Project Tree

```bash
📂 Projeto_Final/
├──assets/
│   ├──Complete_CNN_ConfusionMatrix.png
│   ├──Complete_CNN_Losses.png
│   ├──Simple_CNN_ConfusionMatrix.png
│   └──Simple_CNN_Losses.png
├──models/
│   ├──res_net.py
│   └──simple_cnn.py
├──notebooks/
│   └──tests_notebook.ipynb
├──utils/
│   ├──load_data.py
│   ├──show_data.py
│   └──view_loss.py
├──README.md
├──main.py
└──requirements.txt
```
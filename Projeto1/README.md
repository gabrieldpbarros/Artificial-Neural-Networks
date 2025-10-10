# Rede Neural Supervisionada

## Objetivo

O projeto foi desenvolvido em cima do dataset [csgo_round_snapshots.csv](db/csgo_round_snapshots.csv), que contem aproximadamente 700 exemplos de torneios de alto nível de CS:GO no período entre 2019 e 2020.

A função desta rede neural é ser capaz de prever qual time (Terroristas ou Contraterroristas) vencerá a rodada da partida, jogada em uma melhor de 30 rodadas.

Utilizamos os dados relativos à quantidade de jogadores vivos em cada time, a vida total dos jogadores de cada time, o dinheiro em USD de cada time, o tempo restante no final do round e se a bomba foi plantada, além do time que venceu a rodada.

## Instruções

O projeto consiste em desenvolver um modelo de rede neural supervisionado (rede MLP) para um problema de classificação ou de regressão. Considerar:

- Modelos com diversos hiperparâmetros (número de camadas, número de neurônios, etc.) devem ser avaliados (mínimo de 5 modelos distintos);

- O modelo deve ser treinado com o algoritmo de retropropagação tradicional (SGD - Stochastic Gradient Descendent);

- O dataset deve ser não trivial (não-linear). Considere datasets com pelo menos 5 atributos (features) e no mínimo 200 exemplos;

- O conjunto deve ser dividido entre treino, validação e teste. Considerar 80/10/10%, respectivamente. Aleatorizar antes de separar os conjuntos;

- O modelo com melhor resultado deve retreinado com o termo de momentum.

## Teoria

Para o desenvolvimento da rede, será utilizado o método de **[Correção de Erro](#correção-de-erro)** sob o paradigma do **[Aprendizado Supervisionado](#aprendizado-supervisionado)**, conforme informado nas instruções do projeto.

### Correção de Erro

Os dados (dataset) são fornecidos à rede neural no formato $\{(x_k, d_k)\}^N$, em que $x_k$ são features e $d_k$ são labels (respostas esperadas). Esse formato de dataset cumpre o papel do professor no método de aprendizado, por fornecer as respostas desejadas. A rede, por sua vez, produz uma resposta $y_k$ de acordo com o dado de entrada.

A partir das respostas ($d_k$ e $y_k$), calculamos um sinal de erro ($e_k$) a fim de reduzir o erro ao longo do processo de treinamento:

$$ e_k(n) = d_k(n) - y_k(n) $$

Sendo **n** o número do passo da atualização dos pesos no processo de aprendizado.

Conforme orientado nas instruções do projeto, utilizamos o **gradiente descendente** para processar o erro encontrado e reduzí-lo na próxima iteração do aprendizado.

#### **Gradiente Descendente**

Podemos representar a função de custo, ou energia, pelo erro quadrático médio sobre o conjunto de treinamento:

$$ E(n) = \frac{1}{2}\sum(d_k(n) - y_k(n))^2 $$

ou

$$ E(n) = \frac{1}{2}\sum(e_k(n))^2 $$

Para minimizar essa função, ou seja, reduzir o erro a cada iteração, aplicamos o gradiente descendente, ou regra delta, definido por:

$$ \Delta w_{kj} = -\eta \frac{\partial{E}}{\partial{w_{kj}}} $$

Em que $w_{kj}$ é o peso entre os neurônios k e j, e $\eta$ é a taxa de aprendizagem da rede.

Esse método nos garante a aproximação de um mínimo na função pelo cálculo do negativo do valor do gradiente da função, que indica uma direção de minimização da função.

Vale ressaltar que esse método, por si só, não garante que chegaremos a um mínimo global. Seria necessária a aplicação de etapas extras, como o **termo de momentum**, para garantir que podemos sair de mínimos locais e encontrar mínimos globais.

### Aprendizado Supervisionado

## Dataset

## Desenvolvimento
# Projeto de Previsão de *Churn* de Clientes

Projeto de Ciência de Dados focado na previsão de _churn_ em clientes de telecom, com aplicação de modelos de aprendizagem de máquina e otimização orientada a decisões de negócio.

## Problema

A evasão de clientes (_churn_) impacta diretamente a receita recorrente de empresas de telecom. Antecipar quais clientes têm essa maior propensão permite ações preventivas de retenção.

Este projeto tem como objetivo prever o _churn_ e apoiar decisões de retenção baseadas em custo-benefício.

## Metodologia

- Análise exploratória de dados (EDA);
- Pré-processamento de dados (tratamento, normalização e codificação);
- Comparação entre múltiplas abordagens:
  - Regressão Logística (**LR**);
  - K-Vizinhos mais Próximos (**KNN**);
  - Árvores de Decisão (**DT**);
  - Florestas Aleatórias (**RF**);
  - Máquinas de Vetores de Suporte (**SVM**);
- Otimização de hiperparâmetros com busca aleatória e validação cruzada;
- Análise de _threshold_ baseado nos custos desiguais dos erros.

## Diferenciais do Projeto

- Ajuste de threshold baseado em custo de negócio (não apenas métricas padrão)
- Comparação entre múltiplos cenários de custo (R = 2, 5, 10)
- Separação clara entre treinamento, validação e teste (evitando data leakage)
- Pipeline completo de modelagem e inferência

## Resultados

O modelo **LR** apresentou o melhor equilíbrio entre precisão e _recall_ nos cenários de diferentes _thresholds_, além de ser computacionalmente o mais leve e um dos mais interpretáveis. (O modelo **KNN** apresentou melhor equilíbrio na situação de _threshold_ padrão.)

Exemplo de impacto (cenário com R=5):

- aproximadamente 93% dos _churns_ identificados;
- **_trade-off_**: aproximadamente 54% de clientes abordados desnecessariamente.

## Estrutura do Projeto
```
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling_evaluation.ipynb
│   └── 03_model-comparison_threshold-tuning.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   ├── eda.py
│   ├── evaluation.py
│   ├── modeling.py
│   ├── models.py
│   ├── preprocessing.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt.py
```

## Como Executar o Projeto
```
pip install -r requirements.txt
```
e então rodar os notebooks.

## Visualização dos Notebooks
Para uma experiência de visualização completa, se possível, acessar pelo NBViewer (principalmente o notebook 3):

> [Notebook 1: EDA](https://nbviewer.org/github/levyfelipess/projeto_churn-prediction/blob/main/notebooks/01_eda.ipynb) \
> [Notebook 2: Modelagem e Avaliação](https://nbviewer.org/github/levyfelipess/projeto_churn-prediction/blob/main/notebooks/02_modeling_evaluation.ipynb) \
> [Notebook 3: Comparação entre modelos e Análise de *Threshold*](https://nbviewer.org/github/levyfelipess/projeto_churn-prediction/blob/main/notebooks/03_model-comparison_threshold-tuning.ipynb)

## Conclusão
O projeto demonstra como modelos preditivos podem ser ajustados para decisões reais de negócio, considerando custos assimétricos de erro e impacto operacional.

# Projeto de Previsão de *Churn* de Clientes

Projeto de Ciência de Dados focado na previsão de _churn_ em clientes de telecom, com aplicação de modelos de aprendizagem de máquina e otimização orientada a decisões de negócio.

## Problema

A evasão de clientes (_churn_) impacta diretamente a receita recorrente de empresas de telecom. Antecipar quais clientes têm essa maior propensão permite ações preventivas de retenção.

Este projeto tem como objetivo prever o _churn_ e apoiar decisões de retenção baseadas em custo-benefício.

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

## Visualização dos Notebooks
Para uma experiência de visualização completa, se possível, acessar pelo NBViewer (principalmente o Notebook 3):

> [Notebook 1: EDA](https://nbviewer.org/github/levyfelipess/projeto_churn-prediction/blob/main/notebooks/01_eda.ipynb) \
> [Notebook 2: Modelagem e Avaliação](https://nbviewer.org/github/levyfelipess/projeto_churn-prediction/blob/main/notebooks/02_modeling_evaluation.ipynb) \
> [Notebook 3: Comparação entre modelos e Análise de *Threshold*](https://nbviewer.org/github/levyfelipess/projeto_churn-prediction/blob/main/notebooks/03_model-comparison_threshold-tuning.ipynb)

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

## Principais Diferenciais do Projeto

### Organização e Engenharia de Projetos
- Estrutura modular organizada (`src/`, `notebooks/`, `data/`, `models/`, `reports/`);
- Separação entre código reutilizável e notebooks exploratórios;
- Persistência de modelos treinados;
- Funções e classes documentadas;
- Reprodutibilidade e rastreamento dos experimentos (Git);

### EDA
- Teste estatístico $\chi^2$ de associação entre variáveis categóricas de entrada e a saída;
  - Avaliação da intensidade de associação com coeficiente V de Cramér;
- _Insights_ de negócio;

### Pipeline de Aprendizagem de Máquina
- _Data splitting_ preservando a estratificação da classe de saída;
- Comparação entre abordagens lineares, não lineares, paramétricas, não paramétricas;
- Práticas de prevenção total a _data leakage_;
- Otimização de hiperparâmetros com busca aleatória, validação cruzada estratificada, maximização da AUROC e regra "1SE";
- Estratégias de monitoramento e proteção a _overfitting_;
- Avaliação com métricas diversas e foco nas mais adequadas levando em consideração: **1)** problema, e **2)** desbalanceamento do _dataset_;

### Análise de _Threshold_
- Ajuste de _threshold_ baseado na consideração de custos desiguais de negócio (não apenas métricas padrão);
- Comparação entre múltiplos cenários de custo (R = 2, 5, 10);

## Como Executar o Projeto
```
pip install -r requirements.txt
```
e então rodar os notebooks.

## Resultados

O modelo **LR** apresentou o melhor equilíbrio entre precisão e _recall_ nos cenários de diferentes _thresholds_, além de ser computacionalmente um dos mais leves e também mais interpretáveis. (O modelo **KNN** apresentou melhor equilíbrio na situação de _threshold_ padrão.)

Exemplo de impacto (cenário com R=5):

- aproximadamente 93% dos _churns_ identificados;
- **_trade-off_**: aproximadamente 54% de clientes abordados desnecessariamente.

## Conclusão
O projeto demonstra como aprendizagem de máquina pode ser utilizada para decisões reais de negócio, considerando diferença nos custos de erro e impacto na operacionalização.

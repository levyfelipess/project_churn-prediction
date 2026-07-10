# (EN) Customer Churn Prediction Project

Data Science project focused on predicting churn in telecom customers,using machine learning models and business-oriented decision making.

## Problem

Customer churn directly impacts the recurring revenue of telecom companies. Anticipating which customers are most prone to churn allows for preventative retention actions.

This project aims to predict churn and support cost-benefit-based retention decisions.

## Methodology

### Dataset
#### Links
- [Link 1: Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn);
- [Link 2: IBM](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

#### About the dataset
- The data contains information about a fictitious company that provided landline and internet services to 7043 customers in California (source: [Link 2](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113));
- Each row represents a customer, and each column contains that customer's attributes (source: [Link 1](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)):
  - Customers who canceled service in the last month: column `Churn`;
  - Services contracted by each client: telephone (column `PhoneService`), multiple lines (column `MultipleLines`), internet (column `InternetService`), online security (column `OnlineSecurity`), online backup (column `OnlineBackup`), device protection (column `DeviceProtection`), technical support (column `TechSupport`), TV streaming (column `StreamingTV`) and movie streaming (column `StreamingMovies`);
  - Client account information: relationship length (column `tenure`), contract type (column `Contract`), payment method (column `PaymentMethod`), electronic billing (column `PaperlessBilling`), monthly charges (column `MonthlyCharges`) and total amount due (column `TotalCharges`);
  - Demographic information about customers: gender (column `gender`), age range (column `SeniorCitizen`), and whether they have a partner (column `Partner`) or dependents (column `Dependents`).

### Approach
- Exploratory Data Analysis (EDA);
- Data preprocessing (treatment, normalization, and encoding);
- Comparison between multiple approaches:
  - Logistic Regression (LR);
  - K-Nearest Neighbors (KNN);
  - Decision Trees (DT);
  - Random Forests (RF);
  - Support Vector Machines (SVM);
- Hyperparameter optimization with random search and cross-validation;
- Threshold analysis based on unequal error costs.

## Key Project Highlights

### Project Organization and Engineering
- Organized modular structure (`src/`, `notebooks/`, `data/`, `models/`, `reports/`);
- Separation between reusable code and exploratory notebooks;
- Persistence of trained models;
- Documented functions and classes;
- Reproducibility and traceability of experiments (Git);

### EDA
- Statistical test of association between categorical input variables and the output (chi^2);
- Assessment of the strength of association using Cramer's V coefficient;
- Business insights;

### Machine Learning Pipeline
- Data splitting preserving the stratification of the output class;
- Comparison between linear, non-linear, parametric, and non-parametric approaches;
- Practices for total data leakage prevention;
- Hyperparameter optimization with random search, stratified cross-validation, AUROC maximization, and the "1SE" rule;
- Overfitting control;
- Evaluation with several metrics, focusing on the most appropriate ones, considering: **1)** the problem, and **2)** the imbalance of the dataset;

### Threshold Analysis
- Threshold tuning based on consideration of unequal business costs (not just standard metrics);
- Comparison between multiple cost scenarios (R = 2, 5, 10);

## Project Structure
```
project_churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling-and-evaluation.ipynb
│   └── 03_model-comparison_threshold-tuning.ipynb
│
├── reports/
│   ├── figures/
│   │   ├── en/
│   │   └── pt-br/
│   └── tables/
│
├── src/
│   └── churn_proj/
│       ├── eda/
│       ├── evaluation/
│       ├── modeling/
│       ├── preprocessing/
│       └── utils/
│
├── .gitignore
├── CHANGELOG.md
├── README.md
└── pyproject.toml
```

## How to Run the Project
We recommend using *git bash*:

1. Clone the repository:
```
git clone https://github.com/levyfelipess/project_churn-prediction
```
2. Enter the directory:
```
cd project_churn-prediction
```
3. Create a virtual environment:
```
python -m venv .venv
```
4. Activate the virtual environment (git bash):
```
source .venv/Scripts/activate
```
5. Install the complete project (external dependencies and the package under `src/`) in editable mode:
```
pip install -e .
```
6. Create a specific kernel for the environment:
```
python -m ipykernel install --user --name=`kernel-name`
```
Afterwards, select the created kernel when opening the notebooks.

## Notebook Viewing
For a complete viewing experience, if possible, access via NBViewer (especially Notebook 3):

> [Notebook 1: EDA](https://nbviewer.org/github/levyfelipess/project_churn-prediction/blob/main/notebooks/01_eda.ipynb) \
> [Notebook 2: Modeling and Evaluation](https://nbviewer.org/github/levyfelipess/project_churn-prediction/blob/main/notebooks/02_modeling-and-evaluation.ipynb) \
> [Notebook 3: Model Comparison and Threshold Analysis](https://nbviewer.org/github/levyfelipess/project_churn-prediction/blob/main/notebooks/03_model-comparison_threshold-tuning.ipynb)

## Results

**LR** model showed the best balance between precision and recall in scenarios with different thresholds, in addition to being computationally one of the lightest and most interpretable. (The **KNN** model showed a better balance in the standard threshold situation.)

Example of impact (scenario with R=5):

- approximately 93% of churns identified;

- **trade-off**: approximately 54% of customers impacted unnecessarily.

## Conclusion
This project demonstrates how machine learning can be used for real-world business decisions, considering the difference in error costs and impact on operations.

---
# (PT-BR) Projeto de Previsão de *Churn* de Clientes

Projeto de Ciência de Dados focado na previsão de _churn_ em clientes de telecom, com aplicação de modelos de aprendizagem de máquina e otimização orientada a decisões de negócio.

## Problema

A evasão de clientes (_churn_) impacta diretamente a receita recorrente de empresas de telecom. Antecipar quais clientes têm essa maior propensão permite ações preventivas de retenção.

Este projeto tem como objetivo prever o _churn_ e apoiar decisões de retenção baseadas em custo-benefício.

## Metodologia

### _Dataset_
#### Links de acesso
- [Link 1: Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn);
- [Link 2: IBM](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

#### Sobre o _dataset_
- Os dados contêm informações sobre uma empresa fictícia que forneceu serviços de telefonia fixa e internet para 7043 clientes na Califórnia (fonte: [Link 2](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113));
- Cada linha representa um cliente, e cada coluna contém os atributos deste (fonte: [Link 1](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)):
  - Clientes que cancelaram o serviço no último mês: coluna `Churn`;
  - Serviços contratados por cada cliente: telefone (coluna `PhoneService`), múltiplas linhas (coluna `MultipleLines`), internet (coluna `InternetService`), segurança online (coluna `OnlineSecurity`), backup online (coluna `OnlineBackup`), proteção de dispositivos (coluna `DeviceProtection`), suporte técnico (coluna `TechSupport`), streaming de TV (coluna `StreamingTV`) e de filmes (coluna `StreamingMovies`);
  - Informações da conta do cliente: tempo de relacionamento (coluna `tenure`), tipo de contrato (coluna `Contract`), forma de pagamento (coluna `PaymentMethod`), faturamento eletrônico (coluna `PaperlessBilling`), valores mensais (coluna `MonthlyCharges`) e valor total a pagar (coluna `TotalCharges`);
  - Informações demográficas sobre os clientes: sexo (coluna `gender`), faixa etária (coluna `SeniorCitizen`) e se possuem cônjuge (coluna `Partner`) ou dependentes (coluna `Dependents`).

### Abordagem
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

## Estrutura do Projeto
```
project_churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling-and-evaluation.ipynb
│   └── 03_model-comparison_threshold-tuning.ipynb
│
├── reports/
│   ├── figures/
│   │   ├── en/
│   │   └── pt-br/
│   └── tables/
│
├── src/
│   └── churn_proj/
│       ├── eda/
│       ├── evaluation/
│       ├── modeling/
│       ├── preprocessing/
│       └── utils/
│
├── .gitignore
├── CHANGELOG.md
├── README.md
└── pyproject.toml
```

## Como Executar o Projeto
Recomendamos a utilização do *git bash*:

1. Clonar o repositório:
```
git clone https://github.com/levyfelipess/project_churn-prediction
```
2. Acessar o repositório:
```
cd project_churn-prediction
```
3. Criar um ambiente virtual:
```
python -m venv .venv
```
4. Ativar o ambiente virtual (git bash):
```
source .venv/Scripts/activate
```
5. Instalar o projeto completo (dependências externas e o pacote em `src/`) em modo editável:
```
pip install -e .
```
6. Criar um kernel específico para o ambiente virtual:
```
python -m ipykernel install --user --name=`kernel-name`
```
Após, selecionar o kernel criado quando abrir os notebooks.

## Visualização dos Notebooks
Para uma experiência de visualização completa, se possível, acessar pelo NBViewer (principalmente o Notebook 3):

> [Notebook 1: EDA](https://nbviewer.org/github/levyfelipess/project_churn-prediction/blob/main/notebooks/01_eda.ipynb) \
> [Notebook 2: Modelagem e Avaliação](https://nbviewer.org/github/levyfelipess/project_churn-prediction/blob/main/notebooks/02_modeling-and-evaluation.ipynb) \
> [Notebook 3: Comparação entre modelos e Análise de *Threshold*](https://nbviewer.org/github/levyfelipess/project_churn-prediction/blob/main/notebooks/03_model-comparison_threshold-tuning.ipynb)

## Resultados

O modelo **LR** apresentou o melhor equilíbrio entre precisão e _recall_ nos cenários de diferentes _thresholds_, além de ser computacionalmente um dos mais leves e também mais interpretáveis. (O modelo **KNN** apresentou melhor equilíbrio na situação de _threshold_ padrão.)

Exemplo de impacto (cenário com R=5):

- aproximadamente 93% dos _churns_ identificados;
- **_trade-off_**: aproximadamente 54% de clientes abordados desnecessariamente.

## Conclusão
O projeto demonstra como aprendizagem de máquina pode ser utilizada para decisões reais de negócio, considerando diferença nos custos de erro e impacto na operacionalização.

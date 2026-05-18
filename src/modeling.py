import numpy as np
from scipy.stats import sem
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from time import perf_counter
from src.utils import display_elapsed_time

def train_model(model, X, y, display_language='pt-br'):
    """
    (EN)
    Trains a model using '.fit()' and displays the elapsed time during the process.

    Args:
        model: Model with '.fit()' method for training;
        X (np.array): Matrix of training input patterns;
        y (np.array): Vector of training output patterns;
        display_language (str, optional): Display language.
    
    ---
    (PT-BR)
    Treina um modelo com '.fit()' e exibe o tempo decorrido durante um processo.

    Args:
        model: Modelo com método '.fit()' para treino;
        X (np.array): Matriz de padrões de entrada de treino;
        y (np.array): Vetor de padrões de saída de treino;
        display_language (str, optional): Idioma de exibição.
    """
    time0 = perf_counter()
    model.fit(X=X, y=y)
    timef = perf_counter()
    
    display_elapsed_time(final_time=timef, initial_time=time0, display_language=display_language)

def random_search_with_kfoldcv(model,
                               df_X_train, df_y_train,
                               n_comb, k, hyperparam_distributions,
                               transformer_X, transformer_y,
                               global_seed=7, verbose_tqdm=True, verbose_language='pt-br'):
    """
    (EN)
    Performs hyperparameter optimization by maximizing the AUROC using random search with stratified k-fold cross-validation.

    Args:
        model: Model;
        df_X_train (pd.DataFrame): Training dataset with input variables;
        df_y_train (pd.DataFrame or pd.Series): Training dataset with output variable;
        n_comb (int): Number of hyperparameter combinations to be evaluated;
        k (int): Number of cross-validation folds;
        hyperparam_distributions (dict[str, scipy.stats or list]): Collection of hyperparameter distributions;
        transformer_X: Transformation or set of transformations on the input variables;
        transformer_y: Transformation or set of transformations on the output variable;
        global_seed (int, optional): Random seed to be applied in all stochastic algorithms (reproducibility);
        verbose_tqdm (bool, optional): Indicates whether tqdm progress bars should be displayed;
        verbose_language (str, optional): Display language for tqdm progress bars.

    Returns:
        pd.DataFrame: Table with the tested combinations along with their average metrics in the validation sets.

    Notes:
        In "hyperparam_distributions", it is possible to pass a list with the elements to be chosen randomly, and so the selection will be
        uniform.
    
    ---
    (PT-BR)
    Performa otimização de hiperparâmetros por maximização da AUROC utilizando busca aleatória com validação cruzada k-fold estratificada.

    Args:
        model: Modelo;
        df_X_train (pd.DataFrame): Dataset de treinamento com atributos de entrada;
        df_y_train (pd.DataFrame or pd.Series): Dataset de treinamento com atributo de saída;
        n_comb (int): Número de combinações de hiperparâmetros a serem avaliadas;
        k (int): Número de partições da validação cruzada;
        hyperparam_distributions (dict[str, scipy.stats or list]): Coleção de distribuições dos hiperparâmetros;
        transformer_X: Transformação ou conjunto de transformações nas variáveis de entrada;
        transformer_y: Transformação ou conjunto de transformações na variável de saída;
        global_seed (int, optional): Semente aleatória a ser aplicada em todos os algoritmos estocásticos (reprodutibilidade);
        verbose_tqdm (bool, optional): Indica se barras de progresso do tqdm devem ser mostradas;
        verbose_language (str, optional): Idioma de exibição das barras de progresso tqdm.

    Returns:
        pd.DataFrame: Tabela com as combinações testadas juntamente com suas métricas médias nos conjuntos de validação.

    Notes:
        Em "hyperparam_distributions", é possível passar uma lista com os elementos a serem escolhidos aleatoriamente, e então a escolha
        será de modo uniforme.
    """
    if verbose_language=='pt-br':
        tqdm_desc = 'Validação Cruzada K-Fold'
    elif verbose_language=='en':
        tqdm_desc = 'K-Fold Cross Validation'
    else:
        raise NotImplementedError
    masks = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=global_seed).split(df_X_train, df_y_train))
    hyperparam_combinations = {}
    for key in hyperparam_distributions.keys():
        if type(hyperparam_distributions[key]) == list:
            rng = np.random.default_rng(seed=global_seed)
            hyperparam_combinations[key] = rng.choice(hyperparam_distributions[key], size=n_comb)
        else:
            hyperparam_combinations[key] = hyperparam_distributions[key].rvs(size=n_comb, random_state=global_seed)

    metrics_hyperparams = np.empty((k, n_comb))

    original_model = model
    for i in range(k):
        model = original_model
        
        est_mask, val_mask = masks[i]
        df_X_est, df_y_est = df_X_train.iloc[est_mask], df_y_train.iloc[est_mask]
        df_X_val, df_y_val = df_X_train.iloc[val_mask], df_y_train.iloc[val_mask]

        X_est_transformed = transformer_X.fit_transform(df_X_est)
        y_est = transformer_y.fit_transform(df_y_est)
        X_val_transformed = transformer_X.transform(df_X_val)
        y_val = transformer_y.transform(df_y_val)

        if verbose_tqdm:
            range_combinations = tqdm(range(n_comb), desc=tqdm_desc+f' - {i+1}/{k}')
        else:
            range_combinations = range(n_comb)
            
        for comb in range_combinations:
            for key in hyperparam_combinations.keys():
                model.__dict__[key] = hyperparam_combinations[key][comb]

            model.fit(X=X_est_transformed,
                      y=y_est.reshape(-1))
            y_val_prob_pred = model.predict_proba(X=X_val_transformed)[:, 1]
        
            metrics_hyperparams[i][comb] = roc_auc_score(y_score=y_val_prob_pred, y_true=y_val)

        del model
            
    hyperparam_combinations['auc (mean)'] = np.mean(metrics_hyperparams, axis=0)
    hyperparam_combinations['auc (std)'] = np.std(metrics_hyperparams, axis=0, ddof=1)
    hyperparam_combinations['auc (se)'] = sem(metrics_hyperparams, axis=0, ddof=1)
    
    return pd.DataFrame(hyperparam_combinations)

def parallelized_random_search_with_kfoldcv(model,
                                            df_X_train, df_y_train,
                                            n_comb, k, hyperparam_distributions,
                                            transformer_X, transformer_y,
                                            n_jobs, global_seed=7, display_language='pt-br'):
    """
    (EN)
    Performs hyperparameter optimization by maximizing the AUROC using random search with stratified k-fold cross-validation and
    parallelization over the folds.

    Args:
        model: Model;
        df_X_train (pd.DataFrame): Training dataset with input variables;
        df_y_train (pd.DataFrame or pd.Series): Training dataset with output variable;
        n_comb (int): Number of hyperparameter combinations to be evaluated;
        k (int): Number of cross-validation folds;
        hyperparam_distributions (dict[str, scipy.stats or list]): Collection of hyperparameter distributions;
        transformer_X: Transformation or set of transformations on the input variables;
        transformer_y: Transformation or set of transformations on the output variable;
        n_jobs: Number of cores;
        global_seed (int, optional): Random seed to be applied in all stochastic algorithms (reproducibility);
        display_language (str, optional): Display language.

    Returns:
        pd.DataFrame: Table with the tested combinations along with their average metrics in the validation sets.

    Notes:
        In "hyperparam_distributions", it is possible to pass a list with the elements to be chosen randomly, and so the selection will be
        uniform.
    
    ---
    (PT-BR)
    Performa otimização de hiperparâmetros por maximização da AUROC utilizando busca aleatória com validação cruzada k-fold estratificada
    com a possibilidade de paralelização nos folds.

    Args:
        model: Modelo;
        df_X_train (pd.DataFrame): Dataset de treinamento com atributos de entrada;
        df_y_train (pd.DataFrame or pd.Series): Dataset de treinamento com atributo de saída;
        n_comb (int): Número de combinações de hiperparâmetros a serem avaliadas;
        k (int): Número de partições da validação cruzada;
        hyperparam_distributions (dict[str, scipy.stats or list]): Coleção de distribuições dos hiperparâmetros;
        transformer_X: Transformação ou conjunto de transformações nas variáveis de entrada;
        transformer_y: Transformação ou conjunto de transformações na variável de saída;
        n_jobs: Número de núcleos;
        global_seed (int, optional): Semente aleatória a ser aplicada em todos os algoritmos estocásticos (reprodutibilidade);
        display_language (str, optional): Idioma de exibição.

    Returns:
        pd.DataFrame: Tabela com as combinações testadas juntamente com suas métricas médias nos conjuntos de validação.

    Notes:
        Em "hyperparam_distributions", é possível passar uma lista com os elementos a serem escolhidos aleatoriamente.
        A escolha será de modo uniforme.
    """
    masks = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=global_seed).split(df_X_train, df_y_train))
    hyperparam_combinations = {}
    for key in hyperparam_distributions.keys():
        if type(hyperparam_distributions[key]) == list:
            rng = np.random.default_rng(seed=global_seed)
            hyperparam_combinations[key] = rng.choice(hyperparam_distributions[key], size=n_comb)
        else:
            hyperparam_combinations[key] = hyperparam_distributions[key].rvs(size=n_comb, random_state=global_seed)

    def parallel_kfoldcv_fn(i):
        metrics_hyperparams = np.empty(n_comb)
        
        est_mask, val_mask = masks[i]
        df_X_est, df_y_est = df_X_train.iloc[est_mask], df_y_train.iloc[est_mask]
        df_X_val, df_y_val = df_X_train.iloc[val_mask], df_y_train.iloc[val_mask]

        X_est_transformed = transformer_X.fit_transform(df_X_est)
        y_est = transformer_y.fit_transform(df_y_est)
        X_val_transformed = transformer_X.transform(df_X_val)
        y_val = transformer_y.transform(df_y_val)

        for comb in range(n_comb):
            for key in hyperparam_combinations.keys():
                model.__dict__[key] = hyperparam_combinations[key][comb]

            model.fit(X=X_est_transformed,
                      y=y_est.reshape(-1))
            y_val_prob_pred = model.predict_proba(X=X_val_transformed)[:, 1]
        
            metrics_hyperparams[comb] = roc_auc_score(y_score=y_val_prob_pred, y_true=y_val)
            
        return metrics_hyperparams
    time0 = perf_counter()
    metrics_hyperparams = np.array( Parallel(n_jobs=n_jobs)(delayed(parallel_kfoldcv_fn)(i) for i in range(k)) )
    timef = perf_counter()
    display_elapsed_time(final_time=timef, initial_time=time0, display_language=display_language)
            
    hyperparam_combinations['auc (mean)'] = np.mean(metrics_hyperparams, axis=0)
    hyperparam_combinations['auc (std)'] = np.std(metrics_hyperparams, axis=0, ddof=1)
    hyperparam_combinations['auc (se)'] = sem(metrics_hyperparams, axis=0, ddof=1)
    
    return pd.DataFrame(hyperparam_combinations)

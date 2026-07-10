from time import perf_counter
from ..utils.utils import display_elapsed_time

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
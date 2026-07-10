import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class ZScoreNormalizer(BaseEstimator, TransformerMixin):
    """
    (EN)
    Z-Score normalization using sample standard deviation.

    Attributes:
        ddof (int): Degrees of freedom in the standard deviation calculation;
        mean_vec (np.array): Vector of means of features;
        std_vec (np.array): Vector of standard deviations of features.

    Notes:
        The parent class BaseEstimator provides compatibility with other methods in sklearn;
        The parent class TransformerMixin automatically adds the ".fit_transform" method.
    
    ---
    (PT-BR)
    Normalização Z-Score que utiliza desvio padrão amostral.

    Attributes:
        ddof (int): Graus de liberdade no cálculo do desvio padrão;
        mean_vec (np.array): Vetor de médias dos atributos;
        std_vec (np.array): Vetor de desvios padrões dos atributos.

    Notes:
        A classe pai BaseEstimator proporciona compatabilidade com outros métodos em sklearn;
        A classe pai TransformerMixin adiciona o método ".fit_transform" automaticamente.
    """
    def __init__(self, ddof=1):
        """
        (EN)
        Initializes the Normalizer.

        Args:
            ddof (int, optional): Degrees of freedom in the standard deviation calculation.
    
        ---
        (PT-BR)
        Inicializa o Normalizador.

        Args:
            ddof (int, optional): Graus de liberdade no cálculo do desvio padrão.
        """
        self.mean_vec = None
        self.std_vec = None
        self.ddof = ddof

    def fit(self, X, y=None):
        """
        (EN)
        Learn the parameters of the normalization.

        Args:
            X (np.array): Matrix of patterns and features, in "n x d" format;
            y (None): Only for compatibility with other methods in sklearn.
    
        ---
        (PT-BR)
        Aprende os parâmetros da normalização.

        Args:
            X (np.array): Matriz de padrões e atributos, no formato "n x d";
            y (None): Apenas para compatibilização com outros métodos em sklearn.
        """
        self.mean_vec = np.mean(X, axis=0)
        self.std_vec = np.std(X, axis=0, ddof=self.ddof)
        return self

    def transform(self, X):
        """
        (EN)
        Apply the normalization.

        Args:
            X (np.array): Matrix of patterns and features, in "n x d" format.
            
        Returns:
            np.array: Normalized matrix "X".
    
        ---
        (PT-BR)
        Aplica a normalização.

        Args:
            X (np.array): Matriz de padrões e atributos, no formato "n x d".

        Returns:
            np.array: Matriz "X" normalizada.
        """
        return (X - self.mean_vec) / self.std_vec

    def inverse_transform(self, X):
        """
        (EN)
        Apply the inverse of normalization.

        Args:
            X (np.array): Normalized matrix of patterns and features, in "n x d" format.
            
        Returns:
            np.array: Matrix "X" with denormalized means and standard deviations.
    
        ---
        (PT-BR)
        Aplica o inverso da normalização.

        Args:
            X (np.array): Matriz de padrões e atributos normalizada, no formato "n x d".

        Returns:
            np.array: Matriz "X" com médias e desvios padrões desnormalizados.
        """
        return (X * self.std_vec) + self.mean_vec

def get_transformers(df_X, df_y):
    """
    (EN)
    Gets the specific transformations needed for the data, for each model.

    Args:
        df_X (pd.DataFrame): Complete dataset of input variables;
        df_y (pd.DataFrame or pd.Series): Complete dataset of output variable.

    Returns:
        dict: Collection of transformations on the output variable, for each model;
        OneHotEncoder: One-hot transformation for the output variable.
    
    ---
    (PT-BR)
    Obtém as transformações específicas necessárias aos dados, para cada modelo.

    Args:
        df_X (pd.DataFrame): Dataset completo dos atributos de entrada;
        df_y (pd.DataFrame or pd.Series): Dataset  completo do atributo de saída.

    Returns:
        dict: Coleção das transformações nos atributos de saída, para cada modelo;
        OneHotEncoder: Transformação one-hot para o atributo de saída.
    """
    
    transformers_X = {
    'LR':ColumnTransformer(
        [('OneHot', OneHotEncoder(sparse_output=False, drop='first'), df_X.dtypes[df_X.dtypes == 'object'].index),
         ('ZScore', ZScoreNormalizer(), df_X.dtypes[df_X.dtypes == 'float64'].index)], remainder='passthrough'),
    'KNN':ColumnTransformer(
        [('OneHot', OneHotEncoder(sparse_output=False, drop='first'), df_X.dtypes[df_X.dtypes == 'object'].index),
         ('ZScore', ZScoreNormalizer(), df_X.dtypes[df_X.dtypes == 'float64'].index)], remainder='passthrough'),
    'SVM':ColumnTransformer(
        [('OneHot', OneHotEncoder(sparse_output=False, drop='first'), df_X.dtypes[df_X.dtypes == 'object'].index),
         ('ZScore', ZScoreNormalizer(), df_X.dtypes[df_X.dtypes == 'float64'].index)], remainder='passthrough'),
    'DT':ColumnTransformer(
        [('OneHot', OneHotEncoder(sparse_output=False), df_X.dtypes[df_X.dtypes == 'object'].index)], remainder='passthrough'),
    'RF':ColumnTransformer(
        [('OneHot', OneHotEncoder(sparse_output=False), df_X.dtypes[df_X.dtypes == 'object'].index)], remainder='passthrough')
    }

    transformer_y = OneHotEncoder(sparse_output=False, drop='first', categories=[['No', 'Yes']])

    return transformers_X, transformer_y
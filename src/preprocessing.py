import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class ZScoreNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, ddof=1):
        self.mean_vec = None
        self.std_vec = None
        self.ddof = ddof

    def fit(self, X, y=None):
        self.mean_vec = np.mean(X, axis=0)
        self.std_vec = np.std(X, axis=0, ddof=self.ddof)
        return self

    def transform(self, X):
        return (X - self.mean_vec) / self.std_vec

    def inverse_transform(self, X):
        return (X * self.std_vec) + self.mean_vec

def clean_data(df):
    df = df.drop(columns='customerID')

    col_missing_values = np.where(df['TotalCharges'] == ' ')[0]
    df.loc[col_missing_values, 'TotalCharges'] = df.loc[col_missing_values, 'MonthlyCharges']

    df = df.astype({'SeniorCitizen':'object', 'tenure':'float64', 'TotalCharges':'float64'})
    return df

def get_transformers(df_X, df_y):
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

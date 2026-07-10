import numpy as np

def clean_data_eda(df):
    """
    (EN)
    Applies all necessary transformations to the original dataset to make it ready for EDA.

    Args:
        df(pd.DataFrame): Original dataset (raw).

    Returns:
        pd.DataFrame: Processed dataset.

    Notes:
        Transformations for the dataset in this problem include:
            1. Removing the "customerID" column;
            2. Imputing missing values ​​in "TotalCharges";
            3. Converting the type of "SeniorCitizen", "tenure", and "TotalCharges".
    
    ---
    (PT-BR)
    Aplica todas as transformações necessárias ao dataset original para torná-lo pronto para eda.

    Args:
        df (pd.DataFrame): Dataset original (raw).

    Returns:
        pd.DataFrame: Dataset tratado (processed).

    Notes:
        As transformações para o dataset deste problema incluem:
            1. Retirada da coluna de ID "customerID";
            2. Preenchimento de valores faltantes em "TotalCharges";
            3. Conversão do tipo de "SeniorCitizen", "tenure" e "TotalCharges".
    """
    df = df.drop(columns='customerID')

    col_missing_values = np.where(df['TotalCharges'] == ' ')[0]
    df.loc[col_missing_values, 'TotalCharges'] = df.loc[col_missing_values, 'MonthlyCharges']

    df = df.astype({'SeniorCitizen':'object', 'tenure':'float64', 'TotalCharges':'float64'})
    return df

def clean_data_modeling(df):
    """
    (EN)
    Applies all necessary transformations to the original dataset to make it ready for modeling.

    Args:
        df(pd.DataFrame): Original dataset (raw).

    Returns:
        pd.DataFrame: Processed dataset with input variables;
        pd.DataFrame: Processed dataset with output variable.

    Notes:
        Transformations for the dataset in this problem include:
            1. Removing the "customerID" column;
            2. Removing the "gender" and "PhoneService" variables;
            3. Imputing missing values ​​in "TotalCharges";
            4. Converting the type of "SeniorCitizen", "tenure", and "TotalCharges".
    
    ---
    (PT-BR)
    Aplica todas as transformações necessárias ao dataset original para torná-lo pronto para modelagem.

    Args:
        df (pd.DataFrame): Dataset original (raw).

    Returns:
        pd.DataFrame: Dataset tratado (processed) com variáveis de entrada;
        pd.DataFrame: Dataset tratado (processed) com variável de saída.

    Notes:
        As transformações para o dataset deste problema incluem:
            1. Retirada da coluna de ID "customerID";
            2. Retirada das variáveis "gender" e "PhoneService";
            2. Preenchimento de valores faltantes em "TotalCharges";
            3. Conversão do tipo de "SeniorCitizen", "tenure" e "TotalCharges".
    """
    df = df.drop(columns=['customerID', 'gender', 'PhoneService'])

    col_missing_values = np.where(df['TotalCharges'] == ' ')[0]
    df.loc[col_missing_values, 'TotalCharges'] = df.loc[col_missing_values, 'MonthlyCharges']

    df = df.astype({'SeniorCitizen':'object', 'tenure':'float64', 'TotalCharges':'float64'})
    df_X = df.drop(columns='Churn')
    df_y = df.loc[:, ['Churn']].copy()
    
    return df_X, df_y

def detect_empty_string(df, empty_string_size_min=0, empty_string_size_max=10):
    """
    (EN)
    Detects and display variables containing empty strings of type '', ' ', ...

    Args:
        df (pd.DataFrame): Complete Dataframe;
        empty_string_size_min (int, optional): Minimum number of spaces in the empty string;
        empty_string_size_max (int, optional): Maximum number of spaces in the empty string.
    
    ---
    (PT-BR)
    Detecta e exibe variáveis contendo strings vazias do tipo '', ' ', ...

    Args:
        df (pd.DataFrame): Dataframe completo;
        empty_string_size_min (int, optional): Número mínimo de espaços na string vazia;
        empty_string_size_max (int, optional): Número máximo de espaços na string vazia;
    """
    features_with_empty_strings = np.array([], dtype=np.int32)
    for size in range(empty_string_size_min, empty_string_size_max + 1):
        features_with_empty_strings = np.concatenate((features_with_empty_strings, np.unique(np.where(df == ' ' * size)[1])))
    if len(features_with_empty_strings) > 0:
        display(df.columns[features_with_empty_strings])
    else:
        print("No empty strings!")
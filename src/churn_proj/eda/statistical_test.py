import numpy as np
from numpy import format_float_positional as ffp
from scipy import stats
import pandas as pd

def cramerv_inputs_target(df, target_column=-1, style_format='standard', significance=0.05, save=False, path='chi2-test_result.csv',
                          display_language='pt-br'):
    """
    (EN)
    Displays a table containing Cramer's V between the output and categorical inputs, as well as important values ​​from the Chi-square test
    of association.

    Args:
        df (pd.DataFrame): Dataframe containing only categorical variables;
        target_column (str or int, optional): Key of the output variable. If int, it will be interpreted as the column position;
        style_format (dict[str, str or function] or str=='standard', optional): Formatting style. If style_format=='standard',
                                                                                uses the already programmed style;
        significance (float, optional): Significance level of the Chi-square test.
        save (bool, optional): Indicates whether the dataframe should be saved;
        path (str, optional): Complete storage path;
        display_language (str, optional): Display language.

    Notes:
        Using df.loc[:, df.dtypes=='object'] to select categorical variables usually works;
        Table with interpretation values ​​for Cramer's V available at: https://www.statology.org/interpret-cramers-v/
    
    ---
    (PT-BR)
    Exibe tabela contendo o V de Cramér entre a saída e as entradas categóricas, bem como valores importantes do teste Chi2 de associação.
    
    Args:
        df (pd.DataFrame): Dataframe contendo somente variáveis categóricas;
        target_column (str or int, optional): Chave da variável de saída. Se int, será interpretado como a posição da coluna;
        style_format (dict[str, str or function] or str=='standard', optional): Estilo de formatação. Se style_format=='standard',
                                                                                utiliza o estilo já programado;
        significance (float, optional): Nível de significância do teste Chi2.
        save (bool, optional): Indica se o dataframe deve ser salvo;
        path (str, optional): Caminho completo de armazenamento;
        display_language (str, optional): Idioma de exibição.

    Notes:
        Utilizar df.loc[:, df.dtypes=='object'] para selecionar variáveis categóricas geralmente funciona;
        Tabela com valores de interpretação do V de Cramér disponível em: https://www.statology.org/interpret-cramers-v/
    """
    cramerv_interpretation_table = pd.DataFrame({
            'ddof':[1,2,3,4,5],
            'weak':[.10,.07,.06,.05,.04],
            'moderate':[.30,.21,.17,.15,.13],
            'strong':[.50,.35,.29,.25,.22]
        }).set_index('ddof')
    
    d = df.shape[1]
    if type(target_column) == int:
        target_column = df.columns[target_column]

    if display_language=='pt-br':
        df_caption = "V de Cramér entre entradas e a variável '"+target_column+"'"
    elif display_language=='en':
        df_caption = "Cramer's V between inputs and the variable '"+target_column+"'"
    else:
        raise NotImplementedError
    
    if style_format == 'standard':
        style_format = {'V':lambda x:ffp(x, 4, min_digits=4),
                        'Chi2': lambda x:ffp(x, 2),
                        'p-value': lambda x:ffp(x, 3),
                        'Degrees of Freedom': lambda x:int(x)}
    contingency_df = pd.DataFrame({},
                                  columns=['V', 'Chi2', 'p-value', 'Degrees of Freedom', "Association's Degree"], index=df.columns)

    for i in range(d):
        contingency_table = stats.contingency.crosstab(df.iloc[:, i], df.loc[:, target_column])
        chi2, pvalue, ddof = stats.contingency.chi2_contingency(contingency_table.count)[:3]
        V = stats.contingency.association(observed=contingency_table.count, method='cramer', correction=True)
        contingency_df.iloc[i, 0] = V
        contingency_df.iloc[i, 1] = chi2
        contingency_df.iloc[i, 2] = pvalue
        contingency_df.iloc[i, 3] = ddof
        if pvalue > significance:
            contingency_df.iloc[i, 4] = 'no association'
        else:
            id_degree_association = np.argmin(np.abs(cramerv_interpretation_table.loc[ddof] - V))
            contingency_df.iloc[i, 4] = cramerv_interpretation_table.columns[id_degree_association]

    if save:
        contingency_df.sort_values(by='V', ascending=False).iloc[1:].to_csv(path)

    display(contingency_df.sort_values(
        by='V', ascending=False
        ).iloc[1:].style.set_caption(df_caption).format(style_format)
            )
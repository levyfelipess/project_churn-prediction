from numpy import format_float_positional as ffp
import pandas as pd
from ..utils.utils import highlight_nthmax
from .single_model import evaluate


def evaluate_several_models(models,
                            transformers_X, transformer_y,
                            df_X_train, df_X_test, df_y_train, df_y_test,
                            threshold=0.5, beta_fscore=2.):
    """
    (EN)
    Evaluates several models.

    Args:
        models (dict[str, model]): Models to be evaluated;
        transformers_X (dict[str, transformer]): Normalizations of the input variables, specific to each model;
        transformer_y: Normalization of the output variable;
        df_X_train (pd.DataFrame): Training matrix of the input variables without normalization;
        df_X_test (pd.DataFrame): Test matrix of the input variables without normalization;
        df_y_train (pd.DataFrame or pd.Series): Training vector of the output variable without normalization;
        df_y_test (pd.DataFrame or pd.Series): Test vector of the output variable without normalization;
        threshold (float, optional): Decision threshold for the positive class;
        beta_fscore (float, optional): Weighting of the Recall in the F-beta Score metric;

    Returns:
        dict[str, str]: Collection of metrics in the test set for each model.

    ---
    (PT-BR)
    Avalia diversos modelos.

    Args:
        models (dict[str, model]): Modelos a serem avaliados;
        transformers_X (dict[str, transformer]): Normalizações dos atributos de entrada, específicas para cada modelo;
        transformer_y: Normalização do atributo de saída;
        df_X_train (pd.DataFrame): Matriz de treinamento dos atributos de entrada sem normalização;
        df_X_test (pd.DataFrame): Matriz de teste dos atributos de entrada sem normalização;
        df_y_train (pd.DataFrame or pd.Series): Vetor de treinamento do atributo de saída sem normalização;
        df_y_test (pd.DataFrame or pd.Series): Vetor de teste do atributo de saída sem normalização;
        threshold (float, optional): Limiar de decisão para classe positiva;
        beta_fscore (float, optional): Ponderação do Recall na métrica F-beta Score;

    Returns:
        dict[str, str]: Coleção de métricas no conjunto de testes para cada modelo.
    """
    metrics_final_comparison = {}
    y_train = transformer_y.fit_transform(df_y_train)
    y_test = transformer_y.transform(df_y_test)
    for key in models.keys():
        if key != 'RG' and key != 'IRG':
            X_train_transformed = transformers_X[key].fit_transform(df_X_train)
            X_test_transformed = transformers_X[key].transform(df_X_test)
        else:
            X_train_transformed = df_X_train.copy()
            X_test_transformed = df_X_test.copy()
        
        metrics_final_comparison[key] = evaluate(model=models[key],
                                                 X_train=X_train_transformed, X_test=X_test_transformed, y_train=y_train, y_test=y_test,
                                                 beta_fscore=beta_fscore, threshold=threshold,
                                                 display_metrics_table=False, plot_confusion_matrix=False, plot_roc_pr_curve=False,
                                                 save_metrics_table=False, save_confusion_matrix=False, save_roc_pr_curve=False,
                                                 plot_display_language='pt-br')['Teste']
    return metrics_final_comparison

def display_final_comparison_with_highlight(metrics_final_comparison_dict,
                                            table_title,
                                            beta_fscore=None,
                                            save_table=False,
                                            path_table='final-comparison.csv',
                                            display_language='pt-br'):
    """
    (EN)
    Displays a table with metrics from several models for final comparison, highlighting the:
    - 1st best: dark blue cells (darkblue) and bold letters;
    - 2nd best: light blue cells (steelblue);
    - worst: light red cells (tomato).
    
    Args:
        metrics_final_comparison_dict (dict[str, str or float]): Collection of metrics for each model;
        table_title (str): Table title;
        beta_fscore (float or None, optional): Recall weight used in the Fbeta-Score metric.
                                               If beta_fscore=None, the table will display and store the designation 'Fbeta-Score';
        save_table (bool, optional): Indicates whether the table should be saved;
        path_table (str, optional): Full storage path, if the table is saved;
        display_language (str, optional): Display language.

    ---
    (PT-BR)
    Exibe uma tabela as métricas de diversos modelos para comparação final, com realce para os:
    - 1º melhores: células azuis escuras (darkblue) e letras destacadas;
    - 2º melhores: células azuis claras (steelblue);
    - piores:      células vermelhas claras (tomato).

    Args:
        metrics_final_comparison_dict (dict[str, str or float]): Coleção das métricas para cada modelo;
        table_title (str): Título da tabela;
        beta_fscore (float or None, optional): Peso do Recall utilizado na métrica Fbeta-Score;
                                               se beta_fscore=None, a tabela exibirá e armazenará a designação 'Fbeta-Score'.
        save_table (bool, optional): Indica se a tabela deve ser salva;
        path_table (str, optional): Caminho completo de armazenamento, caso a tabela seja salva;
        display_language (str, optional): Idioma de exibição.
    """
    if beta_fscore==None:
        fbeta_score_str = 'Fbeta-Score'
    else:
        fbeta_score_str = 'F'+ffp(beta_fscore, 2)+'-Score'

    if display_language=='pt-br':
        metrics_index = ['Acurácia','Precisão','Recall','F1-Score',fbeta_score_str,'AUROC', 'AUPR']
    elif display_language=='en':
        metrics_index = ['Accuracy','Precision','Recall','F1-Score',fbeta_score_str,'AUROC', 'AUPR']
        
    metrics_final_comparison_df = pd.DataFrame(metrics_final_comparison_dict,
                                               index=metrics_index)
    if save_table:
        metrics_final_comparison_df.to_csv(path_table)
        
    display(
        metrics_final_comparison_df.style.set_caption(
            table_title
        ).apply(highlight_nthmax(nth_max=1),
            axis=1, props=("color:white; font-weight:bold; background-color:darkblue;")
        ).apply(highlight_nthmax(nth_max=2),
            axis=1, props=("color:white; background-color:steelblue;")
        ).highlight_min(
            axis=1, props=("color:white; font-weight:bold; background-color:tomato;")
        )
    )

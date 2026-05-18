import numpy as np
from numpy import format_float_positional as ffp, format_float_scientific as ffs
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='notebook', style='ticks')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

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

def boxplot(df, feature_name, save=False, path='boxplot.png', plot_language='pt-br', plot=True):
    """
    (EN)
    Plots two custom boxplots: 1) boxplot of the numeric variable without conditioning, and 2) conditioned by the output class.

    Args:
        df (pd.DataFrame): Dataframe containing (at least) the output variable 'Churn' and the variable in 'feature_name';
        feature_name (str): Variable key in the dataframe;
        save (bool, optional): Indicates whether the boxplot should be saved;
        plot (bool, optional): Indicates whether the boxplot should be plotted;
        path (str, optional): Complete storage path;
        plot_language (str, optional): Plotting language.
    
    ---
    (PT-BR)
    Plota dois boxplots personalizados: 1) da variável numérica sem condicionamento e 2) condicionado pela classe de saída.

    Args:
        df (pd.DataFrame): Dataframe contendo (pelo menos) a variável de saída 'Churn' e a variável em questão;
        feature_name (str): Chave da variável no dataframe;
        save (bool, optional): Indica se o boxplot deve ser salvo;
        plot (bool, optional): Indica se o boxplot deve ser plotado;
        path (str, optional): Caminho completo de armazenamento;
        plot_language (str, optional): Idioma de plotagem.
    """
    if plot_language=='pt-br':
        ax_title = ['Geral', 'Condicionado pelas classes de saída']
    elif plot_language=='en':
        ax_title = ['General', 'Conditioned by the output classes']
    else:
        raise NotImplementedError
        
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
    sns.boxplot(ax=ax[0],
                data=df, y=feature_name,
                color=".8", linecolor="#137", linewidth=.75, width=.2, flierprops={'marker':'x'})
    sns.boxplot(ax=ax[1],
                data=df, y=feature_name, x='Churn',
                color=".8", linecolor="#137", linewidth=.75, width=.2, flierprops={'marker':'x'})
    ax[0].set_xticks([])
    fig.suptitle('Boxplots')
    ax[0].set_title(ax_title[0])
    ax[1].set_title(ax_title[1])
    ax[0].grid(lw=.5, axis='y')
    ax[1].grid(lw=.5, axis='y')
    if save:
        fig.savefig(path, dpi=300)
    if plot:
        plt.show()
    plt.close()

def barplot_with_percent(df, feature_name, save=False, path='barplot.png', plot_language='pt-br', plot=True):
    """
    (EN)
    Plots two custom bar charts: 1) bar with frequency of the categorical variable without conditioning, and
                                 2) conditioned by the output class.

    Args:
        df (pd.DataFrame): Dataframe containing (at least) the output variable 'Churn' and the variable in 'feature_name';
        feature_name (str): Variable key in the dataframe;
        save (bool, optional): Indicates whether the bar chart should be saved;
        plot (bool, optional): Indicates whether the bar chart should be plotted;
        path (str, optional): Complete storage path;
        plot_language (str, optional): Plotting language.
    
    ---
    (PT-BR)
    Plota dois gráficos de barras personalizados: 1) frequência da variável categórica em um geral e 2) condicionada pela classe de saída.

    Args:
        df (pd.DataFrame): Dataframe contendo (pelo menos) a variável de saída 'Churn' e a variável em questão;
        feature_name (str): Chave da variável no dataframe;
        save (bool, optional): Indica se o gráfico de barras deve ser salvo;
        plot (bool, optional): Indica se o gráfico de barras deve ser plotado;
        path (str, optional): Caminho completo de armazenamento.
        plot_language (str, optional): Idioma de plotagem.
    """
    if plot_language=='pt-br':
        ax_ylabel = 'Frequência'
        fig_title = 'Distribuição de Categorias'
        ax_title = ['Geral', 'Condicionadas pela classe da saída']
    elif plot_language=='en':
        ax_ylabel = 'Frequency'
        fig_title = 'Category Distribution'
        ax_title = ['General', 'Conditioned by the output classes']
    else:
        raise NotImplementedError
    
    contingency_table = stats.contingency.crosstab(df.loc[:, feature_name], df.loc[:, 'Churn'])
    class_order = contingency_table.elements[0]
    target_class_percent = np.round(contingency_table.count / contingency_table.count.sum(axis=1).reshape(-1, 1) * 100, 1)
    class_percent = np.round(contingency_table.count.sum(axis=1) / contingency_table.count.sum() * 100, 1)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    sns.countplot(ax=ax[0], data=df, x=feature_name, order=class_order, stat='count', color='gray', width=.4)
    sns.countplot(ax=ax[1], data=df, x=feature_name, hue='Churn', order=class_order, stat='count')
    ax[0].bar_label(ax[0].containers[0], labels=class_percent, size=10)
    ax[1].bar_label(ax[1].containers[0], labels=target_class_percent[:, 0], size=10)
    ax[1].bar_label(ax[1].containers[1], labels=target_class_percent[:, 1], size=10)
    ax[0].set_ylabel(ax_ylabel)
    ax[1].set_ylabel(ax_ylabel)
    ax[0].set_xlabel(feature_name)
    ax[1].set_xlabel(feature_name)
    fig.suptitle(fig_title)
    ax[0].set_title(ax_title[0])
    ax[1].set_title(ax_title[1])
    ax[0].grid(lw=.5, axis='y')
    ax[1].grid(lw=.5, axis='y')
    if save:
        fig.savefig(path, dpi=300)
    if plot:
        plt.show()
    plt.close()

def barplot_with_percent_target(df, save=False, path='barplot.png', plot_language='pt-br', plot=True):
    """
    (EN)
    Plots a customized bar chart of frequency for the output variable 'Churn'.

    Args:
        df (pd.DataFrame): Dataframe containing (at least) the output variable 'Churn';
        save (bool, optional): Indicates whether the bar chart should be saved;
        plot (bool, optional): Indicates whether the bar chart should be plotted;
        path (str, optional): Complete storage path;
        plot_language (str, optional): Plotting language.
    
    ---
    (PT-BR)
    Plota gráfico de barras de frequência para a variável de saída 'Churn', personalizado.

    Args:
        df (pd.DataFrame): Dataframe contendo (pelo menos) a variável de saída 'Churn';
        save (bool, optional): Indica se o gráfico de barras deve ser salvo;
        plot (bool, optional): Indica se o gráfico de barras deve ser plotado;
        path (str, optional): Caminho completo de armazenamento;
        plot_language (str, optional): Idioma de plotagem.
    """
    if plot_language=='pt-br':
        ax_ylabel = 'Frequência'
        ax_title = 'Distribuição de Categorias'
    elif plot_language=='en':
        ax_ylabel = 'Frequency'
        ax_title = 'Category Distribution'
    else:
        raise NotImplementedError
    
    target = df.loc[:, 'Churn']
    class_order = target.unique()
    class_percent = np.round(target.value_counts() / target.size * 100, 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
    sns.countplot(ax=ax, x=target, order=class_order, stat='count', color='gray', width=.4)
    ax.bar_label(ax.containers[0], labels=class_percent, size=10)
    ax.set_ylabel(ax_ylabel)
    ax.set_title(ax_title)
    ax.grid(lw=.5, axis='y')
    if save:
        fig.savefig(path, dpi=300)
    if plot:
        plt.show()
    plt.close()

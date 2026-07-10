import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='notebook', style='ticks')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

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

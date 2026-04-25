import sys
import os
sys.path.append(os.path.abspath(".."))
import numpy as np
from scipy.integrate import trapezoid
from numpy import format_float_positional as ffp
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, fbeta_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import highlight_nthmax

sns.set_theme(context='notebook', style='ticks')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

def evaluate(model, X_train, X_test, y_train, y_test,
             threshold=0.5, beta_fscore=1, dig=3, model_title='', threshold_step=0.01,
             display_metrics_table=True, plot_confusion_matrix=True, plot_roc_prc=True,
             save_metrics_table=False, save_confusion_matrix=False, save_roc_prc=False,
             path_metrics_table='metrics_table.csv', path_confusion_matrix='confusion_matrix.png', path_roc_prc='roc_pr_curves.png'):
    y_train_prob_pred = model.predict_proba(X=X_train)[:, 1]
    y_test_prob_pred = model.predict_proba(X=X_test)[:, 1]
    y_train_pred = np.int32(y_train_prob_pred >= threshold)
    y_test_pred = np.int32(y_test_prob_pred >= threshold)

    threshold_vec = np.arange(0., 1. + 0.01, 0.01)[::-1]
    fpr_vec = np.empty_like(threshold_vec)
    tpr_vec = np.empty_like(threshold_vec)
    pre_vec = np.empty_like(threshold_vec)
    for i, thr in enumerate(threshold_vec):
        y_pred = np.int32(y_test_prob_pred >= thr)
        conf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)
        fpr, tpr = conf_matrix[:, 1] / conf_matrix.sum(axis=1)
        fpr_vec[i] = fpr
        tpr_vec[i] = tpr
        pre_vec[i] = precision_score(y_pred=y_pred, y_true=y_test, zero_division=1.)
    
    acc_train = accuracy_score(y_pred=y_train_pred, y_true=y_train)
    pre_train = precision_score(y_pred=y_train_pred, y_true=y_train)
    rec_train = recall_score(y_pred=y_train_pred, y_true=y_train)
    f1s_train = f1_score(y_pred=y_train_pred, y_true=y_train)
    fbs_train = fbeta_score(y_pred=y_train_pred, y_true=y_train, beta=beta_fscore)
    
    acc_test = accuracy_score(y_pred=y_test_pred, y_true=y_test)
    pre_test = precision_score(y_pred=y_test_pred, y_true=y_test)
    rec_test = recall_score(y_pred=y_test_pred, y_true=y_test)
    f1s_test = f1_score(y_pred=y_test_pred, y_true=y_test)
    fbs_test = fbeta_score(y_pred=y_test_pred, y_true=y_test, beta=beta_fscore)
    auroc_test = trapezoid(x=fpr_vec, y=tpr_vec)
    auprc_test = trapezoid(x=tpr_vec, y=pre_vec)

    metrics_dict = {
        'Treinamento':[ffp(acc_train, dig, min_digits=dig), ffp(pre_train, dig, min_digits=dig), ffp(rec_train, dig, min_digits=dig),
                       ffp(f1s_train, dig, min_digits=dig), ffp(fbs_train, dig, min_digits=dig),
                       '-----', '-----'],
        'Teste':[ffp(acc_test, dig, min_digits=dig), ffp(pre_test, dig, min_digits=dig), ffp(rec_test, dig, min_digits=dig),
                 ffp(f1s_test, dig, min_digits=dig), ffp(fbs_test, dig, min_digits=dig),
                 ffp(auroc_test, dig, min_digits=dig), ffp(auprc_test, dig, min_digits=dig)]
    }

    if display_metrics_table:
        display(
            pd.DataFrame(metrics_dict,
                index=['Acurácia','Precisão','Recall','F1-Score','F$_{\\beta}$-Score','AUROC', 'AUPRC']
                        ).style.set_caption('Métricas de Avaliação'+model_title)
                )

    if save_metrics_table:
        df_metrics = pd.DataFrame(metrics_dict, index=['Acurácia','Precisão','Recall','F1-Score','Fbeta-Score','AUROC', 'AUPRC'])
        df_metrics.to_csv(path_metrics_table)

    if plot_confusion_matrix:
        conf_matrix = confusion_matrix(y_pred=y_test_pred, y_true=y_test)
        conf_matrix_normalized_pcol = conf_matrix / conf_matrix.sum(axis=0)
        conf_matrix_normalized_prow = conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1)
        conf_matrix_normalized = conf_matrix / conf_matrix.sum()
        
        fig, ax = plt.subplots(1, 4, figsize=(2.5 * 4, 2.5), layout='constrained')
        sns.heatmap(conf_matrix, ax=ax[0], annot=True, annot_kws={'size':15}, fmt='.0f', cmap='Blues', cbar=False, linecolor='black', linewidths=.01)
        sns.heatmap(conf_matrix_normalized_prow, ax=ax[1], annot=True, annot_kws={'size':15}, fmt='.3f', cmap='Blues', cbar=False, linecolor='black', linewidths=.01, vmin=0., vmax=1.)
        sns.heatmap(conf_matrix_normalized_pcol, ax=ax[2], annot=True, annot_kws={'size':15}, fmt='.3f', cmap='Blues', cbar=False, linecolor='black', linewidths=.01, vmin=0., vmax=1.)
        sns.heatmap(conf_matrix_normalized, ax=ax[3], annot=True, annot_kws={'size':15}, fmt='.3f', cmap='Blues', cbar=False, linecolor='black', linewidths=.01, vmin=0., vmax=1.)
        for i in range(4):
            ax[i].set_xticklabels(['No', 'Yes'])
            ax[i].set_yticklabels(['No', 'Yes'], rotation=0)
            ax[i].set_xlabel('Pred.')
            ax[i].set_ylabel('Obs.')
        ax[0].set_title('Absoluta')
        ax[1].set_title('Relativa por linhas')
        ax[2].set_title('Relativa por colunas')
        ax[3].set_title('Relativa total')
        fig.suptitle('Matrizes de Confusão (Teste)'+model_title)
        if save_confusion_matrix:
            fig.savefig(path_confusion_matrix, dpi=300)
        plt.show()

    if plot_roc_prc:
        threshold_vec = np.arange(0., 1. + threshold_step, threshold_step)[::-1]
        fpr_vec = np.empty_like(threshold_vec)
        tpr_vec = np.empty_like(threshold_vec)
        pre_vec = np.empty_like(threshold_vec)
        for i, thr in enumerate(threshold_vec):
            y_pred = np.int32(y_test_prob_pred >= thr)
            conf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)
            fpr, tpr = conf_matrix[:, 1] / conf_matrix.sum(axis=1)
            fpr_vec[i] = fpr
            tpr_vec[i] = tpr
            pre_vec[i] = precision_score(y_pred=y_pred, y_true=y_test, zero_division=1.)
            
        actual_threshold_id = np.argmin(np.abs(tpr_vec - rec_test))

        fig, ax = plt.subplots(1, 2, figsize=(5 * 2, 5), layout='constrained')        
        sns.lineplot(ax=ax[0], x=fpr_vec, y=tpr_vec, lw=1.5, color='black')
        sns.lineplot(ax=ax[0], x=[0., 1.], y=[0., 1.], lw=.7, ls='--', color='black')
        sns.lineplot(ax=ax[1], x=tpr_vec, y=pre_vec, lw=1.5, color='black')
        sns.lineplot(ax=ax[1], x=[0., 1.], y=[y_test.mean()] * 2, lw=.7, ls='--', color='black')
        sns.scatterplot(ax=ax[0], x=[fpr_vec[actual_threshold_id]], y=[tpr_vec[actual_threshold_id]],
                        marker='o', color='white', edgecolor='black', lw=1.,
                        label=f'Actual ({threshold_vec[actual_threshold_id]:.2f})', zorder=3)
        sns.scatterplot(ax=ax[1], x=[tpr_vec[actual_threshold_id]], y=[pre_vec[actual_threshold_id]],
                        marker='o', color='white', edgecolor='black', lw=1.,
                        label=f'Actual ({threshold_vec[actual_threshold_id]:.2f})', zorder=3)
        ax[0].fill_between(x=fpr_vec, y1=np.zeros_like(tpr_vec), y2=tpr_vec, color='black', alpha=.05)
        ax[1].fill_between(x=tpr_vec, y1=np.zeros_like(tpr_vec), y2=pre_vec, color='black', alpha=.05)
        ax[0].set_xticks(np.arange(0, 1.01, .1))
        ax[0].set_yticks(np.arange(0, 1.01, .1))
        ax[1].set_xticks(np.arange(0, 1.01, .1))
        ax[1].set_yticks(np.arange(0, 1.01, .1))
        ax[0].set_xlim(0. - 0.01, 1. + 0.01)
        ax[0].set_ylim(0. - 0.01, 1. + 0.01)
        ax[1].set_xlim(0. - 0.01, 1. + 0.01)
        ax[1].set_ylim(y_test.mean() - .01, 1. + 0.01)
        ax[0].set_xlabel('FPR')
        ax[0].set_ylabel('TPR')
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precisão')
        ax[0].set_title('ROC'+model_title+'\n$(\\text{AUROC}='+ffp(auroc_test, dig, min_digits=dig)+')$')
        ax[1].set_title('PRC'+model_title+'\n$(\\text{AUPRC}='+ffp(auprc_test, dig, min_digits=dig)+')$')
        ax[0].grid(lw=.5)
        ax[1].grid(lw=.5)
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        if save_roc_prc:
            fig.savefig(path_roc_prc, dpi=300)
        plt.show()

    return metrics_dict

def evaluate_all_models_and_get_final_comparison_dict(models,
                                                      transformers_X, transformer_y,
                                                      df_X_train, df_X_test, df_y_train, df_y_test,
                                                      threshold=0.5, beta_fscore=2.):
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
                                                 display_metrics_table=False, plot_confusion_matrix=False, plot_roc_prc=False,
                                                 save_metrics_table=False, save_confusion_matrix=False, save_roc_prc=False)['Teste']
    return metrics_final_comparison

def display_best_hyperparam_combinations(df,
                                         n_best=1,
                                         with_one_standard_error_rule=True,
                                         metric_mean_column='auc (mean)',
                                         metric_se_column='auc (se)',
                                         style_format={}):
    if with_one_standard_error_rule:
        best_hyperparam_id = np.argmax(df.loc[:, metric_mean_column])
        id_bests = np.where(df.loc[:, metric_mean_column] >= df.loc[best_hyperparam_id, metric_mean_column] - df.loc[best_hyperparam_id, metric_se_column])[0]
        display(df.loc[id_bests].sort_values(by=metric_mean_column, ascending=False).style.format(style_format))
        
    else:
        display(df.sort_values(by=metric_mean_column, ascending=False).head(n_best).style.format(style_format))

def display_final_comparison_with_highlight(metrics_final_comparison_dict,
                                            table_title,
                                            save_table=False,
                                            path_table='final-comparison.csv'):
    metrics_final_comparison_df = pd.DataFrame(metrics_final_comparison_dict,
                                               index=['Acurácia','Precisão','Recall','F1-Score','F$_{\\beta}$-Score','AUROC','AUPRC'])
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

import pyexcel as pe
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


# Função que gera a floresta para inferência e dados de análise utilizando scikit
def floresta(resultados, exames, lista_exames, alvo_inferencia):

    dados_treinamento, dados_teste, result_treinamento, result_teste = train_test_split(exames,
                                                                                        resultados,
                                                                                        test_size=1000,
                                                                                        random_state=60)

    modelo = RandomForestClassifier(n_estimators=300,
                                    bootstrap=True,
                                    max_features='sqrt')
    modelo.fit(dados_treinamento, result_treinamento)

    n_nodes = []
    max_depths = []

    for ind_tree in modelo.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    predicoes_treino = modelo.predict(dados_treinamento)
    predicoes_treino_prob = modelo.predict_proba(dados_treinamento)[:, 1]

    predicoes_floresta = modelo.predict(dados_teste)
    predicoes_floresta_prob = modelo.predict_proba(dados_teste)[:, 1]

    importancias = modelo.feature_importances_
    aux = np.argsort(-1*importancias)[:10]

    print('\n\n************************* ' + alvo_inferencia + ' *************************')
    print(f'Número médio de nós: {int(np.mean(n_nodes))}')
    print(f'Profundidade máxima média das árvores: {int(np.mean(max_depths))}')
    print('\n---Exames mais relevantes para ' + alvo_inferencia + ':')
    for i in aux:
        a = np.where(aux == i)
        indice = a[0]+1
        print(str(indice) + ': ' + lista_exames[i+6])
    print('************************************************************')

    plt.style.use('fivethirtyeight')
    plt.rcParams['font.size'] = 18

    baseline = {
        'recall': recall_score(result_teste.astype(np.int), [1 for _ in range(len(result_teste))]),
        'precision': precision_score(result_teste.astype(np.int), [1 for _ in range(len(result_teste))], zero_division=0),
        'roc': 0.5
    }

    results = {
        'recall': recall_score(result_teste.astype(np.int), predicoes_floresta.astype(np.int)),
        'precision': precision_score(result_teste.astype(np.int), predicoes_floresta.astype(np.int), zero_division=0),
        'roc': roc_auc_score(result_teste.astype(np.int), predicoes_floresta_prob.astype(np.float))
    }

    train_results = {
        'recall': recall_score(result_treinamento.astype(np.int), predicoes_treino.astype(np.int)),
        'precision': precision_score(result_treinamento.astype(np.int), predicoes_treino.astype(np.int), zero_division=0),
        'roc': roc_auc_score(result_treinamento.astype(np.int), predicoes_treino_prob.astype(np.float))
    }

    for metric in ['recall', 'precision', 'roc']:
        print(
            f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

    # Cálculo das taxas de falso e verdadeiro positivo
    base_fpr, base_tpr, _ = roc_curve(result_teste.astype(np.int), [1 for _ in range(len(result_teste))])
    model_fpr, model_tpr, _ = roc_curve(result_teste.astype(np.int), predicoes_floresta_prob.astype(np.float))

    plt.figure(figsize=(10, 8))
    plt.rcParams['font.size'] = 14

    # Acurácia das previsões
    acuracia = 'AUC: ' + str(roc_auc_score(result_teste.astype(np.int), predicoes_floresta_prob.astype(np.float)))

    # Plot das curvas
    plt.plot(base_fpr, base_tpr, 'b', label='base')
    plt.plot(model_fpr, model_tpr, 'r', label='modelo')
    plt.text(0.4, 0, acuracia, fontsize=12)
    plt.legend()
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    titulo = 'ROC - ' + alvo_inferencia
    plt.title(titulo)
    plt.show()


def main():
    dados = pe.get_array(file_name='dataset.xlsx')
    # Lista com os nomes das colunas
    lista_exames = dados[0]

    dados.remove(dados[0])
    dados = np.array(dados).T.tolist()
    # Retira a idade dos pacientes
    dados.remove(dados[1])
    dados.remove(dados[0])

    # Colunas vazias removidas para melhorar precisão - evitando que sejam selecionadas
    dados.remove(dados[97])
    dados.remove(dados[91])
    dados.remove(dados[87])
    dados.remove(dados[79])
    dados.remove(dados[75])
    dados.remove(dados[25])

    # Tratar dados não numéricos e traduzi-los para valores numéricos
    for i in range(0, len(dados)):
        for j in range(0, len(dados[0])):
            if dados[i][j] in ['not_done', 'Não Realizado', '']:
                dados[i][j] = '-1'
            elif dados[i][j] in ['detected', 'positive', 'present', 'normal', 'clear', 'light_yellow',
                                 'Urato Amorfo --+', 'Oxalato de Cálcio +++', 'Oxalato de Cálcio -++',
                                 'Urato Amorfo +++']:
                dados[i][j] = '1'
            elif dados[i][j] in ['not_detected', 'negative', 'absent', 'Ausentes']:
                dados[i][j] = '0'
            elif dados[i][j] in ['yellow', 'lightly_cloudy']:
                dados[i][j] = '2'
            elif dados[i][j] in ['citrus_yellow', 'cloudy']:
                dados[i][j] = '3'
            elif dados[i][j] in ['altered_coloring', 'orange']:
                dados[i][j] = '4'
            elif dados[i][j] in ['<1000']:
                dados[i][j] = '1000'

    # Matriz com dados de resultados e exames
    dados = np.array(dados)

    # Matriz apenas com os exames realizados
    exames = dados[4:]
    exames = np.array(exames).T.tolist()

    # Matriz apenas com os resultados de covid
    exames_covid = dados[0]

    # Matriz apenas com resultados de acompanhamento regular
    acompanhamento_regular = dados[1]

    # Matriz apenas com resultados de ala semi-intensiva
    acompanhamento_semi_intensivo = dados[2]

    # Matriz apenas com resultados de ala intensiva
    acompanhamento_intensivo = dados[3]

    # Matriz apenas com pacientes enviados para casa
    domicilio = []
    for i in range(0, len(dados[0])):
        if acompanhamento_regular[i] == '0' and acompanhamento_semi_intensivo[i] == '0' and acompanhamento_intensivo[i] == '0':
            domicilio.append('1')
        else:
            domicilio.append('0')
    domicilio = np.array(domicilio)

    floresta(exames_covid, exames, lista_exames, 'PCR')
    floresta(acompanhamento_regular, exames, lista_exames, 'Acompanhamento Regular')
    floresta(acompanhamento_semi_intensivo, exames, lista_exames, 'Unidade Semi-Intensiva')
    floresta(acompanhamento_intensivo, exames, lista_exames, 'Unidade Intensiva')
    floresta(domicilio, exames, lista_exames, 'Acompanhamento Domiciliar')


if __name__ == '__main__':
    main()

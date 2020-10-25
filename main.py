import pyexcel as pe
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt


# Função que gera a floresta para inferência
def floresta(resultados, exames, lista_exames, alvo_inferencia):

    dados_treinamento, dados_teste, result_treinamento, result_teste = train_test_split(exames,
                                                                                        resultados,
                                                                                        test_size=(1/3),
                                                                                        random_state=30)

    modelo = RandomForestClassifier(n_estimators=100,
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

    print('************************* ' + alvo_inferencia + ' *************************')
    print(f'Número médio de nós: {int(np.mean(n_nodes))}')
    print(f'Profundidade máxima média das árvores: {int(np.mean(max_depths))}')
    print('\n---Exames mais relevantes para ' + alvo_inferencia + ':')
    for i in aux:
        a = np.where(aux == i)
        indice = a[0]+1
        print(str(indice) + ': ' + lista_exames[i+6])
    print('************************************************************\n\n')

    ax = plt.gca()
    rfc_disp = plot_roc_curve(modelo, dados_teste, result_teste, ax=ax)
    plt.show()


def main():
    dados = pe.get_array(file_name='dataset.xlsx')
    # Lista com os nomes das colunas
    lista_exames = dados[0]

    dados.remove(dados[0])
    dados = np.array(dados).T.tolist()
    dados.remove(dados[0])

    # Tratar dados não numéricos e traduzi-los para valores numéricos
    for i in range(0, len(dados)):
        for j in range(0, len(dados[0])):
            if dados[i][j] in ['not_done', 'Não Realizado', '']:
                dados[i][j] = '0'
            elif dados[i][j] in ['detected', 'positive', 'present', 'normal', 'clear', 'light_yellow',
                                 'Urato Amorfo --+', 'Oxalato de Cálcio +++', 'Oxalato de Cálcio -++',
                                 'Urato Amorfo +++']:
                dados[i][j] = '2'
            elif dados[i][j] in ['not_detected', 'negative', 'absent', 'Ausentes']:
                dados[i][j] = '1'
            elif dados[i][j] in ['yellow', 'lightly_cloudy']:
                dados[i][j] = '3'
            elif dados[i][j] in ['citrus_yellow', 'cloudy']:
                dados[i][j] = '4'
            elif dados[i][j] in ['altered_coloring', 'orange']:
                dados[i][j] = '5'
            elif dados[i][j] in ['<1000']:
                dados[i][j] = '1000'

    # Matriz com dados de resultados e exames
    dados = np.array(dados)

    # Matriz apenas com os exames realizados
    exames = dados[5:]
    exames = np.array(exames).T.tolist()

    # Matriz apenas com os resultados de covid
    exames_covid = dados[1]

    floresta(exames_covid, exames, lista_exames, 'Covid')


if __name__ == '__main__':
    main()

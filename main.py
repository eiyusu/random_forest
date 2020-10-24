import pyexcel as pe
import numpy as np


# Função que lê a planilha com os dados e retorna uma matriz com os valores tratados, todos numéricos
def organiza_matriz():
    dados = pe.get_array(file_name='dataset.xlsx')
    dados.remove(dados[0])
    dados = np.array(dados).T.tolist()
    dados.remove(dados[0])

    # Tratar dados não numéricos e traduzi-los para valores numéricos
    for i in range(0, len(dados)):
        for j in range(0, len(dados[0])):
            if dados[i][j] in ['not_done', 'Não Realizado']:
                dados[i][j] = ''
            elif dados[i][j] in ['detected', 'positive', 'present', 'normal', 'clear', 'light_yellow', 'Urato Amorfo --+', 'Oxalato de Cálcio +++', 'Oxalato de Cálcio -++', 'Urato Amorfo +++']:
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
    dados = np.array(dados)
    return dados


if __name__ == '__main__':
    organiza_matriz()

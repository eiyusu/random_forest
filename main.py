# Biblioteca para ler matriz de dados --- pip install pyexcel
import pyexcel as pe

if __name__ == '__main__':
    dados = pe.get_array(file_name='dataset.xlsx')

    print(dados)


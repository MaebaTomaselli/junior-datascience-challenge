import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PreProcessor_TimeSeries:

    def group_by_hour(self, df, col_date):
        """ Realiza o agrupamento dos dados em período de hora.
        
        arg:
        - df: dataframe com dados
        - col_date: coluna que contém a data

        return:
        - df_hora: conjunto de dados agrupados por hora
        """

        df_hora = df.resample('H').first()
        df_hora.index.names = [col_date]

        return df_hora

    def pre_process_data_train_test(self, df_hora, perc_test):

        # Ordenando os dados cronologicamente (caso eles não estejam ordenados)
        df_hora.sort_index(inplace=True)

        # Definindo a proporção de divisão entre treinamento e teste
        train_ratio = perc_test

        # Calculando o índice de divisão
        split_index = int(train_ratio * len(df_hora))

        # Dividindo os dados em conjuntos de treinamento e teste
        train_data = df_hora.iloc[:split_index]
        test_data = df_hora.iloc[split_index:]

        return train_data, test_data
    
    def create_col_lagged (self, df_lagged, coluna, lag):
        
        """ Função para incluir uma coluna com os valores anteriores para indicar a relação temporal ao algoritmo
        
        arg:
        - df_lagged: conjunto de dados
        - coluna: coluna resposta que será utilizada como referência
        - lag: intervalo de avaliação
        
        return:
        - df_return: conjunto de dados acrescido da coluna com dados do dia seguinte à cada linha
        """
        
        df_return = df_lagged.copy()
        df_return['target'] = df_lagged[coluna].shift(lag)
        df_return = df_return.dropna()
        
        
        return df_return
    
    def train_test_split (self, df_train, df_test, columns_X, column_y):
    
        """
        Função para separar o conjunto de dados em treino e teste de acordo com as colunas que são escolhidas 
        para permanecer no modelo
        
        args:
        - df_train: bloco de dados que será utilizado previamente separado para treino;
        - df_test: bloco de dados que será utilizado previamente separado para teste;
        - columns_X: colunas que serão excluídas no modelo que será treinado;
        - column_y: coluna resposta.
        
        return:
        - X_train: bloco de dados com as colunas para treino;
        - y_train: bloco de dados com a coluna resposta do treino;
        - X_test: bloco de dados com as colunas para teste;
        - y_test: bloco de dados com a coluna resposta do teste.
        """
    
        X_train = df_train.drop(columns = columns_X)
        y_train = df_train[column_y]
        X_test = df_test.drop(columns = columns_X)
        y_test = df_test[column_y]
        
        return X_train, y_train, X_test, y_test
    
    def cut_dataset (self, df_cut, del_value):
    
        """
        Função para excluir o conjunto de linhas que não será usado no modelo
        
        args:
        - df_cut: bloco de dados;
        - del_value: data até a qual os dados não serão utilizados
        
        return:
        - df_trat: bloco de dados com o conjunto de dados selecionado
        """
    
        df_trat = df_cut.loc[del_value:]
        
        return df_trat

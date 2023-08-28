import pandas as pd
import matplotlib.pyplot as plt

class DataExplore:

    def data_info(self, df):

        # Análise exploratória dos dados
        print("Informações dos dados")
        df.info()

        print("Descrição resumida dos dados")
        df.describe()

        print("Avaliação de dados duplicados")
        df.duplicated().value_counts()

    def data_period(self, df):

        #Avaliação do período a ser trabalhado
        min_date = df.index.min()
        max_date = df.index.max()
        print("Menor data: ", min_date)
        print("Maior data: ", max_date)

        #Checando os períodos faltantes
        lista_completa_horas = pd.Series(data=pd.date_range(start=min_date, end=max_date, freq='H')) #Cria uma lista com todas as horas entre o mínimo e o máximo do dataset
        lista_dataset_horas = lista_completa_horas.isin(df.index.values)
        lista_completa_horas[~lista_dataset_horas]
        print("Datas faltantes: ", lista_completa_horas)

    def create_graph_freq(self, df_graf, coluna):
        
        '''
        Gera histogramas
        
        arg:
        - df_graf: conjunto de dados;
        - coluna: coluna que será usada para o gráfico
        '''
        
        plt.figure(figsize=(4,4)) 
        plt.hist(df_graf[coluna], label=coluna)
        plt.axvline(df_graf[coluna].mean(), color='red', linestyle='dashed', linewidth=2, label='Média')
        plt.axvline(df_graf[coluna].median(), color='black', linewidth=2, label='Mediana')
        plt.legend()
        plt.show()


    def create_graph_outlier(df_graf, coluna):
        
        '''
        Gera os gráficos box plot
        
        arg:
        - df_graf: conjunto de dados;
        - coluna: coluna que será usada para o gráfico
        
        '''

        df_graf[coluna].plot(kind='box')
        plt.show()

    def quantity_outlier (self, df, column):

        '''
        Quantidade de outlier
        
        arg:
        - df_out: conjunto de dados;
        - column: coluna que será usada para o gráfico
        
        '''

        Q1 = df[column].quantile(0.25) 
        Q3 = df[column].quantile(0.75) 
        IQR=Q3-Q1
        df_out = df.loc[(df[column] <= (Q1 - 1.5 * IQR)) | (df[column] >= (Q3 + 1.5 * IQR))]
        print('Quantidade de outliers para ', column, ': ', df_out[column].value_counts().sum())
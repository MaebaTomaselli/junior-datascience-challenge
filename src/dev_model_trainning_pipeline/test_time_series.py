from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

class Test_time_series:

    def decompose_serie (self, df_decompose, column):

        """
        Faz a decomposição da série temporal

        args:
        - df_decompose: conjunto de dados que será decomposto;
        - column: coluna de avaliação
        
        """


        decompose = seasonal_decompose(df_decompose["column"], model='additive')

        plt.figure(figsize=(10, 6))

        # Plot dos componentes
        plt.subplot(4, 1, 1)
        plt.plot(decompose.observed, label='Observado')
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(decompose.trend, label='Tendência')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(decompose.seasonal, label='Sazonalidade')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(decompose.resid, label='Resíduos')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def tests_stationarity(df, column):

        """
        Avalia a estacionariedade da série

        args: 
        - df: conjunto de dados a ser analisado;
        - column: coluna analisada
        """

        # Selecionando a coluna de interesse para o teste de estacionariedade
        series = df[column]

        # Realizando o teste de Dickey-Fuller Aumentado
        result = adfuller(series)

        # Extraindo os resultados do teste
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]

        print(f'ADF Statistic: {adf_statistic}')
        print(f'P-value: {p_value}')
        print('Critical Values:')
        for key, value in critical_values.items():
            print(f'   {key}: {value}')
            
        if p_value <= 0.05:
            print("Resultado: A série é estacionária")
        else:
            print("Resultado: A série não é estacionária")
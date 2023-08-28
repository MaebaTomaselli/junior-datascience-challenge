#!/usr/bin/env python
# coding: utf-8

# Quality Prediction in a Mining Process

# https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process

#Bibliotecas

#Manipulação de dados
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform

#Gráficos
import seaborn as sns
import matplotlib.pyplot as plt
#from pandas.plotting import autocorrelation_plot

#Séries temporais
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Modelos 
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

#Métricas de avaliação
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Otimização do modelo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#Classes do modelo
import Loader
import PreProcessor_TimeSeries
import DataExplore
import Test_time_series
import Evaluate_model
import Model_train

# Instanciação das classes
loader = Loader()
data_explore = DataExplore()
pre_processor = PreProcessor_TimeSeries()
test_time_serie = Test_time_series()
evaluate = Evaluate_model()
model = Model_train()

# Parâmetros
url_file = "../data/MiningProcess_Flotation_Plant_Database.csv"
index = 'date'

def main():
    # Execução do passos de desenvolvimento

    # Carga
    df = loader.load_data(url_file, index) 

    # Exploração dos dados
    df.data_explore.data_info()
    df.data_explore.data_visuals()

    # Gráfico de avaiação da variável target
    sns.set_palette('Accent')
    sns.set_style('darkgrid')
    df['% Silica Concentrate'].plot(figsize=(10,6))
    plt.title('% Silica Concentrate período completo')
    plt.legend()

    df.data_period()

    # Exclusão dos valores faltantes
    df_trat = pre_processor.cut_dataset(df, '2017-03-29 12:00:00')

    # Avaliação da quantidade de dados por coluna
    # Contagem de valores agrupados por data
    print("Quantidade de dados por agrupamento de segundos", df_trat.groupby(df_trat.index).count()["% Silica Concentrate"].value_counts())

    # Gráfico para a checagem a distribuição de frequência de registros para todas as variáveis
    valores_unicos_hora = df_trat.groupby('date').nunique().mean()
    plt.figure(figsize=(14,8))
    sns.lineplot(x = valores_unicos_hora.index, y=valores_unicos_hora.values, lw=1)
    ax = sns.barplot(x = valores_unicos_hora.index, y=valores_unicos_hora.values)
    plt.xticks(rotation=90)
    plt.title('Frequência de registros com agrupamento por hora')
    for c in ax.containers:
            labels = [f"{round(h,0):.0f}" if (h := v.get_height()) > 0 else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')

    # Avaliando o número de registros por dia por variável
    valores_unicos_dia = df_trat.groupby(df_trat.index.strftime('%Y-%m-%d')).nunique().mean()
    plt.figure(figsize=(14,6),dpi=100)
    sns.lineplot(x = valores_unicos_dia.index, y=valores_unicos_dia.values, lw=1)
    ax = sns.barplot(x = valores_unicos_dia.index, y=valores_unicos_dia.values)
    plt.title("Frequência de registros com agrupamento por dia")
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    for c in ax.containers:
            labels = [f"{round(h,0):.0f}" if (h := v.get_height()) > 0 else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge', **{"size":12})
    
    # Variáveis registradas a cada 20 segundos teriam 4.320 registros em um dia
    freq_registro_silica_conc = df_trat.groupby(df_trat.index.strftime('%Y-%m-%d %H')).nunique()["% Silica Concentrate"]
    freq_registro_silica_conc.value_counts()

    # Transformando os dados para que fiquem em período de hora e não mais em segundos
    df_hora = pre_processor.group_by_hour(df_trat, 'date')

    # Exploração dos dados agrupados por hora
    df_hora.data_explore.data_info()

    # Gráfico com a variável target agrupada
    df_hora['% Silica Concentrate'].plot(figsize=(10,6))
    plt.legend()

    # Separação em bloco de treino e teste
    train_data, test_data = pre_processor.pre_process_data_train_test(df_hora, 0.8) 

    # Análise das variáveis individualmente
    columns = train_data.columns

    for column in columns:
        data_explore.create_graph_freq(train_data, column)

    for column in columns:
        data_explore.create_graph_outlier(train_data, column)

    # Quantidade de outliers para a concentração de sílica
    data_explore.quantity_outlier(train_data, '% Silica Concentrate')

    # Quantidade de outliers para a concentração de ferro
    data_explore.quantity_outlier(train_data, '% Iron Concentrate')

    # Decomposição da série temporal
    test_time_serie.decompose_serie(train_data, '% Silica Concentrate')
    test_time_serie.tests_stationarity(train_data, '% Silica Concentrate')

    # 1º modelo - Random Forest - Aplicação do modelo para intervalo de 1 hora

    # Separação das variáveis e das respostas
    train_data_lagged = pre_processor.create_col_lagged(train_data_lagged, '% Silica Concentrate', -1)
    test_data_lagged = pre_processor.create_col_lagged(test_data_lagged, '% Silica Concentrate', -1)

    X_train, y_train, X_test, y_test = pre_processor.train_test_split(train_data_lagged, 
                                                                        test_data_lagged, 
                                                                        ['target'], 
                                                                        ['target'])
    
    
    random_forest_model, y_pred = model.model_random_forest(X_train, y_train, X_test)

    # Avaliação do modelo
    metricas_random_forest = evaluate.regression_model_evaluation(y_test, y_pred)

    for metric, value in metricas_random_forest.items():
        print(f'{metric}: {value}')

    # Seleção de features
    feature_importances = random_forest_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    
    # Ordenando as features por importância
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print('Sequência de importância das variáveis (Random Forest): ', importance_df)


    # 2º modelo - Random Forest - Teste com as variáveis que tem influência e exclusão das que são diretamente relacionadas

    # Separação dos dados de acordo com as variáveis selecionadas
    X_train_2, y_train_2, X_test_2, y_test_2 = pre_processor.train_test_split(train_data_lagged, 
                                                                                test_data_lagged, 
                                                                                ['% Iron Feed', 
                                                                                '% Silica Feed', 
                                                                                'Amina Flow', 
                                                                                'Ore Pulp Flow', 
                                                                                'Flotation Column 01 Air Flow', 
                                                                                'Flotation Column 02 Air Flow',
                                                                                'Flotation Column 03 Air Flow', 
                                                                                'Flotation Column 04 Air Flow',
                                                                                'Flotation Column 05 Air Flow', 
                                                                                'Flotation Column 06 Air Flow',
                                                                                'Flotation Column 07 Air Flow', 
                                                                                'Flotation Column 01 Level',
                                                                                'Flotation Column 02 Level', 
                                                                                'Flotation Column 03 Level',
                                                                                'Flotation Column 04 Level', 
                                                                                'Flotation Column 05 Level',
                                                                                'Flotation Column 06 Level', 
                                                                                'Flotation Column 07 Level',
                                                                                '% Iron Concentrate', 
                                                                                '% Silica Concentrate'], 
                                                                                ['target'])

    random_forest_model2, y_pred_2 = model.model_random_forest(X_train_2, y_train_2, X_test_2)

    # Avaliação do modelo
    metricas_random_forest2 = evaluate.regression_model_evaluation(y_test_2, y_pred_2)

    for metric, value in metricas_random_forest2.items():
        print(f'{metric}: {value}')

    # 3º modelo - Random Forest com alteração de hiperparâmetros

    # Usando os parâmetros escolhidos para criar o modelo
    random_forest_model5 = RandomForestRegressor(max_depth=3, random_state=42)

    # Treinando o modelo final com os dados de treinamento
    random_forest_model5.fit(X_train_2, y_train_2)

    # Fazendo previsões nos dados de teste
    y_pred_5 = random_forest_model5.predict(X_test_2)

    metricas_random_forest4 = evaluate.regression_model_evaluation(y_test_2, y_pred_5)
    for metric, value in metricas_random_forest4.items():
        print(f'{metric}: {value}')

    # 4º modelo - XGBoost - Todas as variáveis
    xgb_model, y_pred = model.model_random_forest(X_train, y_train, X_test)

    # Avaliação do modelo
    metricas_random_forest = evaluate.regression_model_evaluation(y_test, y_pred)

    for metric, value in metricas_random_forest.items():
        print(f'{metric}: {value}')

    # Aumentando o intervalo (lag) para previsão de 10 horas - usando no treinamento variáveis selecionadas
    train_data_lagged10 = pre_processor.create_col_lagged(train_data_lagged, '% Silica Concentrate', -1)
    test_data_lagged10 = pre_processor.create_col_lagged(test_data_lagged, '% Silica Concentrate', -1)

    X_train_lag10, y_train_lag10, X_test_lag10, y_test_lag10 = pre_processor.train_test_split(train_data_lagged, 
                                                                                                test_data_lagged, 
                                                                                                ['% Iron Feed', 
                                                                                                '% Silica Feed', 
                                                                                                'Amina Flow', 
                                                                                                'Ore Pulp Flow', 
                                                                                                'Flotation Column 01 Air Flow', 
                                                                                                'Flotation Column 02 Air Flow',
                                                                                                'Flotation Column 03 Air Flow', 
                                                                                                'Flotation Column 04 Air Flow',
                                                                                                'Flotation Column 05 Air Flow', 
                                                                                                'Flotation Column 06 Air Flow',
                                                                                                'Flotation Column 07 Air Flow', 
                                                                                                'Flotation Column 01 Level',
                                                                                                'Flotation Column 02 Level', 
                                                                                                'Flotation Column 03 Level',
                                                                                                'Flotation Column 04 Level', 
                                                                                                'Flotation Column 05 Level',
                                                                                                'Flotation Column 06 Level', 
                                                                                                'Flotation Column 07 Level',
                                                                                                '% Iron Concentrate', 
                                                                                                '% Silica Concentrate'], 
                                                                                                ['target'])
    
    
    random_forest_model10, y_pred_lag10 = model.model_random_forest(X_train_lag10, y_train_lag10, X_test_lag10)

    # Avaliação do modelo
    metricas_random_forest = evaluate.regression_model_evaluation(y_test_lag10, y_pred_lag10)

    for metric, value in metricas_random_forest.items():
        print(f'{metric}: {value}')

    # Export do modelo treinado
    #loaded_pkl_model = export_model.export_best_model(best_pipeline)

if __name__ == '__main__':
    main()
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Evaluate_model:

    def regression_model_evaluation (self, y_test_metric, y_pred_metric):
        
        '''
        Calcula e retorna as métricas de avaliação para um modelo de regressão
        
        args:
        - y_test_metric: valores reais (teste)
        - y_pred_metric: valores previstos pelo modelo
        
        Retorna um dicionário com as métricas calculadas
        '''
        
        metrics = {}
        
        metrics['Mean Squared Error'] = mean_squared_error(y_test_metric, y_pred_metric)
        metrics['Mean Absolute Error'] = mean_absolute_error(y_test_metric, y_pred_metric)
        metrics['Root Mean Squared Error'] = np.sqrt(metrics['Mean Squared Error'])
        metrics['R-squared (R²)'] = r2_score(y_test_metric, y_pred_metric)
        
        return metrics
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class Model_train:

    def model_random_forest(self, X_train, y_train, X_test):

        """
        Treinamento do algoritmo Random Forest

        args:
        - X_train: conjunto de variáveis treino;
        - y_train: conjunto de treino com a resposta;
        - X_test: conjunto de variáveis de teste;
        - depth: nível de profundidade da árvore.

        return:
        - random_forest_model: regressor;
        - y_pred: variável predita.
        """

        # Criando o modelo Random Forest
        random_forest_model = RandomForestRegressor(random_state=42)

        # Treinando o modelo
        random_forest_model.fit(X_train, y_train)

        # Fazendo previsões nos dados de teste
        y_pred = random_forest_model.predict(X_test)

        return random_forest_model, y_pred
    
    
    def model_xgboost(self, X_train, y_train, X_test, depth):

        """
        Treinamento do algoritmo XGBoost

        args:
        - X_train: conjunto de variáveis treino;
        - y_train: conjunto de treino com a resposta;
        - X_test: conjunto de variáveis de teste;
        - depth: nível de profundidade da árvore.

        return:
        - xgb_model: regressor;
        - y_pred: variável predita.
        """

        # Criando o modelo Random Forest
        xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)

        # Treinando o modelo
        xgb_model.fit(X_train, y_train)

        # Fazendo previsões nos dados de teste
        y_pred = xgb_model.predict(X_test)

        return xgb_model, y_pred
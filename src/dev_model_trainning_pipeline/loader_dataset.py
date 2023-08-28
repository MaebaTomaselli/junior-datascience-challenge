#Classe retirada de: https://medium.com/joguei-os-dados/largando-os-cadernos-de-notebooks-%C3%A0-scripts-py-20273a1e6629

import pandas as pd

class Loader_index_date:

    def load_data(self, url: str, index_col: str):
        """ Carrega o arquivo e retorna um DataFrame com o índice data e separador ','

        args:
        - url: string com  o nome/endereço do arquivo
        - index_col: string com o nome da coluna que será utilizada como índice
        """  
        return pd.read_csv(url, decimal=',', index_col=index_col, parse_dates=True)
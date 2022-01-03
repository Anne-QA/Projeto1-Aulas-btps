import time
import json
from loguru import logger
from service.constants import mensagens
import pandas as pd


class ContadorPalavrasService():

    def __init__(self):
        logger.debug(mensagens.INICIO_LOAD_CONTADOR)
        self.load_model()

    def load_model(self):
        """"
        Carrega o contador de palavras
        """

        logger.debug(mensagens.FIM_LOAD_CONTADOR)

    def executar_rest(self, texts):
        response = {}

        logger.debug(mensagens.INICIO_PREDICT)
        start_time = time.time()

        response_predicts = self.buscar_predicao(texts['textoMensagem'])

        logger.debug(mensagens.FIM_PREDICT)
        logger.debug(f"Fim de todas as predições em {time.time()-start_time}")

        df_response = pd.DataFrame(texts, columns=['textoMensagem'])
        df_response['predict'] = response_predicts

        df_response = df_response.drop(columns=['textoMensagem'])

        response = {
            "listaClassificacoes": json.loads(df_response.to_json(
                orient='records', force_ascii=False))}

        return response

    def buscar_predicao(self, texts):
        """
        Pega o modelo carregado e aplica em texts
        """
        logger.debug('Iniciando a contagem...')

        response = []

        for text in texts:
            palavras = text.split()
            tamanho = len(palavras)

            response.append(str(tamanho))

        return response

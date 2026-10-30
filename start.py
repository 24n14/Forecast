from config import SYMBOL, DB_DSN
from bybit_client import BybitClient
from db_layer import DatabaseLayer
from fsm import TradingFSM

if __name__ == "__main__":
    client = BybitClient()
    db = DatabaseLayer(DB_DSN)
    fsm = TradingFSM(client, db, SYMBOL, None) # Конфиг можно передать, если нужны динамические параметры
    fsm.run()

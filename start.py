# start.py
from config import SYMBOL, TIMEFRAME, DB_DSN, HISTORY_DAYS
from bybit_client import BybitClient
from db_layer import DatabaseLayer
from fsm import TradingFSM

if __name__ == "__main__":
    print(f"🚀 Запуск бота: {SYMBOL}, TF={TIMEFRAME}")

    client = BybitClient()
    db = DatabaseLayer(DB_DSN)

    # ✅ Передаём реальные значения, никаких None
    fsm = TradingFSM(
        client=client,
        db=db,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        history_days=HISTORY_DAYS
    )

    fsm.run()

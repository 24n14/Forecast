# start.py
from config import (
    SYMBOL, TIMEFRAME, DB_DSN, HISTORY_DAYS,
    NTFY_TOPIC, ENABLE_NOTIFICATIONS,
    SIGNAL_THRESHOLD_STRONG, SIGNAL_THRESHOLD_WEAK
)
from bybit_client import BybitClient
from db_layer import DatabaseLayer
from fsm import TradingFSM

if __name__ == "__main__":
    print(f"🚀 Запуск бота: {SYMBOL}, TF={TIMEFRAME}")

    client = BybitClient()
    db = DatabaseLayer(DB_DSN)

    fsm = TradingFSM(
        client=client,
        db=db,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        history_days=HISTORY_DAYS,
        ntfy_topic=NTFY_TOPIC,
        enable_notifications=ENABLE_NOTIFICATIONS,
        signal_threshold_strong=SIGNAL_THRESHOLD_STRONG,
        signal_threshold_weak=SIGNAL_THRESHOLD_WEAK
    )

    fsm.run()

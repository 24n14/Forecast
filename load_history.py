# load_history.py
# Разовая загрузка полной истории свечей для обучения модели (по таймфрейму из config.py)
from config import SYMBOL, TIMEFRAME, DB_DSN, HISTORY_START
from bybit_client import BybitClient
from db_layer import DatabaseLayer


def main():
    print(f"📥 Загрузка истории {SYMBOL} TF={TIMEFRAME} с {HISTORY_START}...")
    df = BybitClient().fetch_history(SYMBOL, TIMEFRAME, HISTORY_START)
    if df.empty:
        raise SystemExit("❌ Bybit вернул пустую историю")

    db = DatabaseLayer(DB_DSN)
    db.save_candles(df, SYMBOL, TIMEFRAME)

    saved = db.get_all_data(SYMBOL, TIMEFRAME)
    print(f"✅ В базе {len(saved)} свечей: {saved['time'].min()} — {saved['time'].max()}")
    gaps = saved['time'].diff().dropna()
    print(f"   Самый частый шаг: {gaps.mode().iloc[0]}, пропусков (шаг больше обычного): {(gaps > gaps.mode().iloc[0]).sum()}")


if __name__ == "__main__":
    main()

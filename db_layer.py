from sqlalchemy import create_engine, text
import pandas as pd


class DatabaseLayer:
    def __init__(self, dsn):
        self.engine = create_engine(dsn)

        with self.engine.connect() as conn:
            # Создаем таблицу, если её нет
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS candles (
                    time TIMESTAMP PRIMARY KEY,
                    symbol TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL
                )
            """))

            # Создаем индекс для ускорения поиска по времени
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_time ON candles(time)"))
            conn.commit()

    def save_candles(self, df):
        if df.empty:
            return

        # Преобразуем время в строку перед сохранением
        df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        with self.engine.connect() as conn:
            for _, row in df.iterrows():
                conn.execute(text("""
                    INSERT OR REPLACE INTO candles (time, symbol, open, high, low, close, volume)
                    VALUES (:time, :symbol, :open, :high, :low, :close, :volume)
                """), {
                    "time": row["time"],
                    "symbol": row["symbol"] if "symbol" in row else "BTCUSDT",
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })
            conn.commit()

    def get_all_data(self, symbol):
        """Вся история по символу в хронологическом порядке (для обучения модели)."""
        query = text("SELECT * FROM candles WHERE symbol = :symbol ORDER BY time ASC")
        return pd.read_sql(query, self.engine, params={"symbol": symbol}, parse_dates=['time'])

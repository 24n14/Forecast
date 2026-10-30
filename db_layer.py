from sqlalchemy import create_engine, text
import pandas as pd


class DatabaseLayer:
    def __init__(self, dsn):
        self.engine = create_engine(dsn)

    def save_candles(self, df):
        # Создаем таблицу, если нет (для простоты примера)
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS candles (
                    time TIMESTAMPTZ PRIMARY KEY,
                    symbol TEXT,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION
                )
            """))
            conn.commit()

        # Upsert: если время есть, обновляем, иначе вставляем
        # Это быстрее, чем цикл по строкам
        df.to_sql('candles', self.engine, if_exists='append', index=False)

        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO candles (time, symbol, open, high, low, close, volume)
                SELECT time, symbol, open, high, low, close, volume FROM candles
                ON CONFLICT (time) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                    close = EXCLUDED.close, volume = EXCLUDED.volume
            """))
            conn.commit()

    def get_data(self, symbol, days=180):
        query = f"""
            SELECT * FROM candles 
            WHERE symbol = '{symbol}' 
            ORDER BY time DESC LIMIT {days}
        """
        return pd.read_sql(query, self.engine)

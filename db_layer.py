from sqlalchemy import create_engine, text
import pandas as pd


class DatabaseLayer:
    def __init__(self, dsn):
        self.engine = create_engine(dsn)

        with self.engine.connect() as conn:
            columns = [row[1] for row in conn.execute(text("PRAGMA table_info(candles)"))]

            # Миграция: старая таблица без таймфрейма (все свечи дневные)
            if columns and 'interval' not in columns:
                conn.execute(text("ALTER TABLE candles RENAME TO candles_old"))
                conn.execute(text("DROP INDEX IF EXISTS idx_time"))
                self._create_table(conn)
                conn.execute(text("""
                    INSERT INTO candles (time, symbol, interval, open, high, low, close, volume)
                    SELECT time, symbol, 'D', open, high, low, close, volume FROM candles_old
                """))
                conn.execute(text("DROP TABLE candles_old"))
                print("🔄 База обновлена: добавлен таймфрейм свечей (старые свечи помечены как 'D')")
            else:
                self._create_table(conn)

            conn.commit()

    @staticmethod
    def _create_table(conn):
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS candles (
                time TIMESTAMP,
                symbol TEXT,
                interval TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, interval, time)
            )
        """))

    def save_candles(self, df, symbol, interval):
        if df.empty:
            return

        rows = [
            {
                "time": t.strftime('%Y-%m-%d %H:%M:%S'),
                "symbol": symbol,
                "interval": interval,
                "open": float(o), "high": float(h), "low": float(l),
                "close": float(c), "volume": float(v),
            }
            for t, o, h, l, c, v in zip(df['time'], df['open'], df['high'],
                                        df['low'], df['close'], df['volume'])
        ]

        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT OR REPLACE INTO candles (time, symbol, interval, open, high, low, close, volume)
                VALUES (:time, :symbol, :interval, :open, :high, :low, :close, :volume)
            """), rows)
            conn.commit()

    def get_all_data(self, symbol, interval):
        """Вся история по символу и таймфрейму в хронологическом порядке."""
        query = text("SELECT * FROM candles WHERE symbol = :symbol AND interval = :interval ORDER BY time ASC")
        return pd.read_sql(query, self.engine, params={"symbol": symbol, "interval": interval},
                           parse_dates=['time'])

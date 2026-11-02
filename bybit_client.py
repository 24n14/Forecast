# bybit_client.py
from pybit.unified_trading import HTTP
import pandas as pd

class BybitClient:
    def __init__(self):
        self.client = HTTP(testnet=False)

    def fetch_candles(self, symbol, interval, limit=1000):
        try:
            resp = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            if resp.get("retCode") != 0:
                raise Exception(f"Bybit API error: {resp.get('retMsg')}")

            data = resp["result"]["list"]

            # Создаем DataFrame
            df = pd.DataFrame(
                data,
                columns=[
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "turnover"
                ]
            )

            # Преобразуем время в два этапа
            df["time"] = df["time"].astype(int)  # Сначала в int
            df["time"] = pd.to_datetime(df["time"], unit="ms")  # Затем в datetime

            # Приведение к float
            numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
            df[numeric_cols] = df[numeric_cols].astype(float)

            return df

        except Exception as e:
            print(f"❌ Ошибка при загрузке свечей: {e}")
            raise

from pybit.unified_trading import HTTP
import pandas as pd

class BybitClient:
    def __init__(self):
        # testnet=False для продакшена
        self.client = HTTP(testnet=False)

    def fetch_candles(self, symbol, interval, limit=1000):
        resp = self.client.query_kline(symbol=symbol, interval=interval, limit=limit)
        data = resp["result"]["list"]
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        return df

# bybit_client.py
import time

from pybit.unified_trading import HTTP
import pandas as pd

from config import TIMEFRAME_HOURS

MAX_LIMIT = 1000  # Максимум свечей за один запрос Bybit


class BybitClient:
    def __init__(self):
        self.client = HTTP(testnet=False)

    def _request(self, symbol, interval, limit, end=None):
        params = dict(category="linear", symbol=symbol, interval=interval, limit=limit)
        if end is not None:
            params["end"] = end

        resp = self.client.get_kline(**params)
        if resp.get("retCode") != 0:
            raise Exception(f"Bybit API error: {resp.get('retMsg')}")

        df = pd.DataFrame(
            resp["result"]["list"],
            columns=["time", "open", "high", "low", "close", "volume", "turnover"],
        )
        df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms")
        numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        return df

    @staticmethod
    def _drop_unclosed(df, interval):
        """Убирает текущую, ещё не закрытую свечу."""
        candle_len = pd.Timedelta(hours=TIMEFRAME_HOURS[interval])
        now = pd.Timestamp.utcnow().tz_localize(None)
        return df[df["time"] + candle_len <= now]

    def fetch_candles(self, symbol, interval, limit=MAX_LIMIT):
        """Последние `limit` закрытых свечей (до 1000)."""
        try:
            df = self._request(symbol, interval, min(limit, MAX_LIMIT))
            return self._drop_unclosed(df, interval).sort_values("time").reset_index(drop=True)
        except Exception as e:
            print(f"❌ Ошибка при загрузке свечей: {e}")
            raise

    def fetch_history(self, symbol, interval, start):
        """Все закрытые свечи начиная с даты `start` (постранично по 1000)."""
        start_ts = pd.Timestamp(start)
        parts, end = [], None

        while True:
            df = self._request(symbol, interval, MAX_LIMIT, end=end)
            if df.empty:
                break
            parts.append(df)
            oldest = df["time"].min()
            if oldest <= start_ts or len(df) < MAX_LIMIT:
                break
            end = int(oldest.value // 10**6) - 1  # мс, на 1 мс раньше самой старой свечи
            time.sleep(0.1)  # Бережём лимиты API

        if not parts:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "turnover"])

        df = pd.concat(parts).drop_duplicates("time")
        df = df[df["time"] >= start_ts]
        return self._drop_unclosed(df, interval).sort_values("time").reset_index(drop=True)

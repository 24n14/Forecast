# ta_engine.py
import talib as ta

def add_features(df):
    if len(df) < 25:
        # Для SMA(20) + небольшой запас. Можно поднять до 50, если планируешь длинные окна.
        raise ValueError(f"Недостаточно данных для расчёта индикаторов: всего {len(df)} строк, нужно минимум 25.")

    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    o = df['open'].values

    # Индикаторы
    df['sma_20'] = ta.SMA(c, timeperiod=20)
    df['rsi_14'] = ta.RSI(c, timeperiod=14)

    # Паттерны
    df['pattern_engulfing'] = ta.CDLENGULFING(o, h, l, c)
    df['pattern_hammer'] = ta.CDLHAMMER(o, h, l, c)

    # Бинарный флаг
    df['is_bullish_pattern'] = ((df['pattern_engulfing'] == 100) | (df['pattern_hammer'] == 100)).astype(int)

    return df

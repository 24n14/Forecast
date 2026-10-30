import talib as ta
import numpy as np


def add_features(df):
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    o = df['open'].values

    # Индикаторы (берем самые сильные, чтобы не раздувать DF)
    df['sma_20'] = ta.SMA(c, timeperiod=20)
    df['rsi_14'] = ta.RSI(c, timeperiod=14)

    # Паттерны (возвращают 100, -100, 0)
    df['pattern_engulfing'] = ta.CDLENGULFING(o, h, l, c)
    df['pattern_hammer'] = ta.CDLHAMMER(o, h, l, c)

    # Бинарный флаг бычьего паттерна
    df['is_bullish_pattern'] = ((df['pattern_engulfing'] == 100) | (df['pattern_hammer'] == 100)).astype(int)

    return df

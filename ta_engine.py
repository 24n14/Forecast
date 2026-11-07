# ta_engine.py
# Единый источник признаков: индикаторы на свечах (add_features) и агрегация окна (aggregate_window).
# Используется и при обучении (model_trainer.py), и при прогнозе (forecaster.py).
import numpy as np
import talib as ta

# MACD(12, 26, 9) требует 33 свечи прогрева, плюс запас
MIN_ROWS = 40

BULLISH = 100
BEARISH = -100


def add_features(df):
    if len(df) < MIN_ROWS:
        raise ValueError(f"Недостаточно данных для расчёта индикаторов: всего {len(df)} строк, нужно минимум {MIN_ROWS}.")

    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    o = df['open'].values.astype(float)
    v = df['volume'].values.astype(float)

    # --- Индикаторы ---
    df['sma_20'] = ta.SMA(c, timeperiod=20)
    df['rsi_14'] = ta.RSI(c, timeperiod=14)
    df['atr_14'] = ta.ATR(h, l, c, timeperiod=14)

    macd, macd_signal, macd_hist = ta.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

    bb_upper, bb_middle, bb_lower = ta.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower

    df['obv'] = ta.OBV(c, v)

    # --- Свечные паттерны ---
    engulfing = ta.CDLENGULFING(o, h, l, c)        # +100 бычье / -100 медвежье
    df['pattern_engulfing'] = engulfing
    df['pattern_hammer'] = ta.CDLHAMMER(o, h, l, c)                # бычий
    df['pattern_shooting_star'] = ta.CDLSHOOTINGSTAR(o, h, l, c)   # медвежий
    df['pattern_hanging_man'] = ta.CDLHANGINGMAN(o, h, l, c)       # медвежий
    df['pattern_evening_star'] = ta.CDLEVENINGSTAR(o, h, l, c)     # медвежий

    # Бинарные флаги
    df['is_bullish_pattern'] = ((engulfing == BULLISH) | (df['pattern_hammer'] == BULLISH)).astype(int)
    df['is_bearish_pattern'] = (
        (engulfing == BEARISH)
        | (df['pattern_shooting_star'] == BEARISH)
        | (df['pattern_hanging_man'] == BEARISH)
        | (df['pattern_evening_star'] == BEARISH)
    ).astype(int)

    return df


def _trend(values):
    """Корреляция ряда со временем: -1 (падает) ... +1 (растёт)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.std(values) == 0:
        return 0.0
    return float(np.corrcoef(np.arange(len(values)), values)[0, 1])


def aggregate_window(window_df, candles_per_day=1):
    """Агрегирует окно свечей (после add_features) в словарь признаков.

    candles_per_day: свечей в одном дне (D=1, 4h=6) — чтобы "неделя" и "3 дня" означали календарные дни.
    """
    week = 7 * candles_per_day
    three_days = 3 * candles_per_day
    c = window_df['close'].values
    first_close = c[0]
    last_close = c[-1]
    last = window_df.iloc[-1]

    atr_pct = window_df['atr_14'] / window_df['close'] * 100
    bb_width = (window_df['bb_upper'] - window_df['bb_lower']) / window_df['bb_middle'] * 100
    bb_range = last['bb_upper'] - last['bb_lower']
    macd_hist_pct = window_df['macd_hist'] / window_df['close'] * 100

    obv = window_df['obv'].values
    vol_7d = window_df['volume'].iloc[-week:].sum()

    features = {
        # Динамика цены
        'return': (last_close / first_close - 1) * 100,
        'volatility': np.std(np.diff(np.log(c))) * np.sqrt(len(c)) * 100,
        'max_drawdown': (c.min() / c.max() - 1) * 100,
        'price_range_pct': (c.max() / c.min() - 1) * 100,

        # Объём
        'avg_volume': window_df['volume'].mean(),
        'volume_std': window_df['volume'].std(),
        'volume_trend': _trend(window_df['volume']),

        # RSI и SMA
        'avg_rsi': window_df['rsi_14'].mean(),
        'rsi_min': window_df['rsi_14'].min(),
        'rsi_max': window_df['rsi_14'].max(),
        'rsi_last': window_df['rsi_14'].iloc[-1],
        'close_to_sma20_mean': (window_df['close'] / window_df['sma_20'] - 1).mean() * 100,
        'close_to_sma20_last': (last_close / window_df['sma_20'].iloc[-1] - 1) * 100,

        # ATR: текущая волатильность и её режим (расширение/сжатие относительно окна)
        'atr_pct_last': atr_pct.iloc[-1],
        'atr_ratio': atr_pct.iloc[-1] / atr_pct.mean(),

        # MACD (в % от цены, чтобы не зависеть от уровня цены)
        'macd_pct_last': last['macd'] / last_close * 100,
        'macd_hist_pct_last': macd_hist_pct.iloc[-1],
        'macd_hist_slope_3d': macd_hist_pct.iloc[-1] - macd_hist_pct.iloc[-(three_days + 1)],

        # Полосы Боллинджера: положение цены (%B) и ширина (сжатие перед движением)
        'bb_pctb_last': (last_close - last['bb_lower']) / bb_range if bb_range > 0 else 0.5,
        'bb_width_last': bb_width.iloc[-1],
        'bb_width_ratio': bb_width.iloc[-1] / bb_width.mean(),

        # OBV: тренд за окно и чистый поток объёма за неделю (-1 ... +1)
        'obv_trend': _trend(obv),
        'obv_flow_7d': (obv[-1] - obv[-(week + 1)]) / vol_7d if vol_7d > 0 else 0,

        # Паттерны
        'bullish_pattern_count': window_df['is_bullish_pattern'].sum(),
        'bullish_pattern_pct': window_df['is_bullish_pattern'].mean() * 100,
        'bearish_pattern_count': window_df['is_bearish_pattern'].sum(),
        'pattern_balance_7d': (window_df['is_bullish_pattern'].iloc[-week:].sum()
                               - window_df['is_bearish_pattern'].iloc[-week:].sum()),

        # Тренд
        'price_trend': _trend(c),

        # Последняя неделя
        'last_week_return': (c[-1] / c[-(week + 1)] - 1) * 100,
        'last_week_vol': np.std(np.diff(np.log(c[-(week + 1):]))) * np.sqrt(week) * 100,

        # Последние 3 дня
        'last_3d_return': (c[-1] / c[-(three_days + 1)] - 1) * 100,
    }

    return features

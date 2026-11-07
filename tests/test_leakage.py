# tests/test_leakage.py
# Проверка, что признаки не подглядывают в будущее.
# Запуск: python tests/test_leakage.py  (или pytest tests/)
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vol_model import har_frame, HAR_COLS, train_set  # noqa: E402
from ta_engine import add_features, aggregate_window  # noqa: E402
from votes import compute_votes, VOTE_COLS  # noqa: E402

CPD, H = 6, 42


def synthetic_candles(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        'time': pd.date_range('2022-01-01', periods=n, freq='4h'),
        'open': close, 'high': close * 1.005, 'low': close * 0.995,
        'close': close, 'volume': rng.uniform(100, 200, n),
    })


def features_unaffected_by_future(make_features, cols, cut=1500):
    """Портим все свечи после `cut` (×999) — признаки строк <= cut должны совпасть побайтно."""
    df = synthetic_candles()
    spoiled = df.copy()
    spoiled.loc[cut + 1:, ['open', 'high', 'low', 'close', 'volume']] *= 999
    a = make_features(df)[cols].iloc[:cut + 1]
    b = make_features(spoiled)[cols].iloc[:cut + 1]
    return np.allclose(a.values, b.values, equal_nan=True)


def test_har_features_have_no_lookahead():
    assert features_unaffected_by_future(lambda d: har_frame(d, CPD, H), HAR_COLS)


def test_indicator_features_have_no_lookahead():
    # Индикаторы LightGBM (SMA, RSI, ATR, MACD, Боллинджер, OBV, паттерны)
    cols = ['sma_20', 'rsi_14', 'atr_14', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_lower',
            'obv', 'is_bullish_pattern', 'is_bearish_pattern']
    assert features_unaffected_by_future(lambda d: add_features(d.copy()), cols)


def test_window_aggregate_has_no_lookahead():
    # Признаки окна, оканчивающегося на свече cut, не зависят от свечей после неё
    cut, window = 1500, 180
    df = synthetic_candles()
    spoiled = df.copy()
    spoiled.loc[cut + 1:, ['open', 'high', 'low', 'close', 'volume']] *= 999
    a = aggregate_window(add_features(df.copy()).iloc[cut - window + 1:cut + 1], CPD)
    b = aggregate_window(add_features(spoiled.copy()).iloc[cut - window + 1:cut + 1], CPD)
    assert all(np.isclose(a[k], b[k], equal_nan=True) for k in a)


def test_votes_have_no_lookahead():
    # Голоса из старых ботов: ZigZag и MIDAS особенно легко сделать "подглядывающими"
    assert features_unaffected_by_future(lambda d: compute_votes(d, CPD), VOTE_COLS + ['vote_equal'])


def test_method_catches_leaky_feature():
    # Мета-тест: признак с shift(-1) обязан быть пойман
    def leaky(d):
        F = har_frame(d, CPD, H)
        F['leak'] = F['close'].shift(-1)
        return F
    assert not features_unaffected_by_future(leaky, ['leak'])


def test_train_set_purges_future_targets():
    F = har_frame(synthetic_candles(), CPD, H)
    for origin in (800, 1200, 1900):
        tr = train_set(F, origin, H)
        assert tr.index.max() + H <= origin
        assert not tr['ret'].isna().any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f"✓ {name}")
    print("Все тесты пройдены")

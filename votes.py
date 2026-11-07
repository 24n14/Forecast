# votes.py
# Голосование индикаторов — идеи из старых ботов автора, перенесённые на 4h-свечи Forecast_bot.
#
# Каждый голос — число от −1 (медвежий) до +1 (бычий), вычисляется ТОЛЬКО по свечам <= t:
#   ma_trend  — MA-CROSS-bot: угол MA/EMA, нормированный на ATR, расхождение MA/EMA, фильтр флэта по ADX
#   htf       — MA-CROSS-bot: фильтр старшего ТФ (цена относительно дневной средней)
#   midas     — signal_bot_madas: цена относительно MIDAS (VWAP от главного разворота окна)
#   zigzag    — signal_bot_madas: направление последнего подтверждённого сегмента ZigZag
#   breakout  — breakdown-retest: пробой/удержание 30-дневного максимума (минимума)
#   rsi       — signal_bot_madas: перепроданность/перекупленность RSI (контртрендовый голос)
#
# Запуск проверки на истории: python votes.py
import numpy as np
import pandas as pd
import talib as ta

VOTE_COLS = ['ma_trend', 'htf', 'midas', 'zigzag', 'breakout', 'rsi']


def _angle(series, atr, lookback, scale=50.0):
    """Угол наклона в градусах, нормированный на ATR (как в MA-CROSS-bot/indicator.py)."""
    change = series - series.shift(lookback)
    norm = np.clip(change / (atr * lookback) * scale, -10, 10)
    return pd.Series(np.degrees(np.arctan(norm)), index=series.index).rolling(3).mean()


def vote_ma_trend(df, ma_period=21, ema_period=49, lookback=20,
                  weak=3.0, strong=8.0, adx_flat=20.0, div_th=0.15):
    """Направление × уверенность из determine_trend_strength() MA-CROSS-bot (параметры его 4h-конфига)."""
    c, h, l = df['close'], df['high'], df['low']
    atr = pd.Series(ta.ATR(h.values, l.values, c.values, 14), index=df.index)
    adx = pd.Series(ta.ADX(h.values, l.values, c.values, 14), index=df.index)
    ma = c.rolling(ma_period).mean()
    ema = c.ewm(span=ema_period, adjust=False).mean()
    a_ma, a_ema = _angle(ma, atr, lookback), _angle(ema, atr, lookback)
    div = (ma - ema) / ema * 100

    out = np.zeros(len(df))
    for i in range(len(df)):
        am, ae, d, x = a_ma.iloc[i], a_ema.iloc[i], div.iloc[i], adx.iloc[i]
        if np.isnan(am) or np.isnan(ae) or np.isnan(x) or np.isnan(d):
            continue
        if x < adx_flat or max(abs(am), abs(ae)) < weak:
            # Флэт: голос только по расхождению MA/EMA
            conf = 0.5 if x < adx_flat else 0.4
            out[i] = conf * np.sign(d) if abs(d) > div_th else 0.0
            continue
        dom = am if abs(am) >= abs(ae) else ae
        if abs(dom) <= weak:
            continue
        direction = np.sign(dom)
        conf = 0.9 if (abs(am) + abs(ae)) / 2 >= strong else 0.5
        if x >= 50:
            conf += 0.1
        if abs(d) > div_th:
            conf += 0.1 if np.sign(d) == direction else -0.2
        out[i] = direction * float(np.clip(conf, 0.1, 1.0))
    return out


def vote_htf(df, cpd, days=31):
    """Старший ТФ: цена выше/ниже дневной SMA(31) (дневной конфиг MA-CROSS-bot)."""
    sma = df['close'].rolling(days * cpd).mean()
    return np.sign(df['close'] - sma).fillna(0).values


def vote_midas(df, window, deviation_pct=1.0):
    """Цена относительно MIDAS (VWAP) от главного разворота последних `window` свечей."""
    c, h, l, v = (df[k].values.astype(float) for k in ('close', 'high', 'low', 'volume'))
    typical = (h + l + c) / 3
    pv, vv = np.r_[0, np.cumsum(typical * v)], np.r_[0, np.cumsum(v)]
    out = np.zeros(len(df))
    for t in range(window - 1, len(df)):
        lo = t - window + 1
        hi_i, lo_i = lo + np.argmax(h[lo:t + 1]), lo + np.argmin(l[lo:t + 1])
        mid = np.median(c[lo:t + 1])
        anchor = hi_i if h[hi_i] / mid - 1 >= 1 - l[lo_i] / mid else lo_i
        midas = (pv[t + 1] - pv[anchor]) / (vv[t + 1] - vv[anchor])
        if c[t] > midas * (1 + deviation_pct / 100):
            out[t] = 1.0
        elif c[t] < midas * (1 - deviation_pct / 100):
            out[t] = -1.0
    return out


def vote_zigzag(df, depth=5):
    """Направление последнего сегмента ZigZag. Экстремум в свече i известен только в свече i + depth."""
    h, l = df['high'].values, df['low'].values
    n = len(df)
    pivots = []          # (i, type, value) — после фильтра повторов, как в _filter_zigzag_extremes
    out = np.zeros(n)
    state = 0.0
    for t in range(n):
        i = t - depth    # кандидат, чьё окно [i−depth, i+depth] только что закрылось
        if i >= depth:
            lo, hi = i - depth, i + depth + 1
            for typ, val, is_ext in ((1, h[i], h[i] == h[lo:hi].max()), (-1, l[i], l[i] == l[lo:hi].min())):
                if not is_ext:
                    continue
                if pivots and pivots[-1][1] == typ:
                    if (typ == 1 and val > pivots[-1][2]) or (typ == -1 and val < pivots[-1][2]):
                        pivots[-1] = (i, typ, val)
                else:
                    pivots.append((i, typ, val))
            if len(pivots) >= 2:
                (_, t_last, v_last), (_, _, v_prev) = pivots[-1], pivots[-2]
                state = 1.0 if (t_last == 1 and v_last > v_prev) else -1.0 if (t_last == -1 and v_last < v_prev) else 0.0
        out[t] = state
    return out


def vote_breakout(df, cpd, level_days=30, hold_days=7):
    """Пробой 30-дневного максимума (минимума), удерживаемый до 7 дней (идея breakdown-retest)."""
    c, h, l = df['close'], df['high'], df['low']
    w = level_days * cpd
    level_hi = h.shift(1).rolling(w).max()
    level_lo = l.shift(1).rolling(w).min()
    broke_up = (c > level_hi).astype(float)
    broke_dn = (c < level_lo).astype(float)
    recent_up = broke_up.rolling(hold_days * cpd, min_periods=1).max()
    recent_dn = broke_dn.rolling(hold_days * cpd, min_periods=1).max()
    # Пробой засчитан, пока цена не вернулась под пробитый уровень
    hold_hi = level_hi.where(broke_up == 1).ffill()
    hold_lo = level_lo.where(broke_dn == 1).ffill()
    up = (recent_up == 1) & (c > hold_hi)
    dn = (recent_dn == 1) & (c < hold_lo)
    return (up.astype(float) - dn.astype(float)).values


def vote_rsi(df, period=14):
    """RSI: < 30 — бычий голос (перепроданность), > 70 — медвежий."""
    rsi = ta.RSI(df['close'].values.astype(float), period)
    return np.where(rsi < 30, 1.0, np.where(rsi > 70, -1.0, 0.0))


def compute_votes(df, cpd):
    """Все голоса для каждой свечи + итог голосования с равными весами (подход старых ботов)."""
    df = df.sort_values('time').reset_index(drop=True)
    V = pd.DataFrame({'time': df['time'].values})
    V['ma_trend'] = vote_ma_trend(df)
    V['htf'] = vote_htf(df, cpd)
    V['midas'] = vote_midas(df, window=30 * cpd)
    V['zigzag'] = vote_zigzag(df)
    V['breakout'] = vote_breakout(df, cpd)
    V['rsi'] = vote_rsi(df)
    V['vote_equal'] = V[VOTE_COLS].mean(axis=1)
    V['warm'] = np.arange(len(V)) >= 31 * cpd
    return V


# ---------------------------------------------------------------------------
# Проверка на истории: несёт ли голосование информацию о направлении?
# ---------------------------------------------------------------------------
def _hit_stats(vote, up, pi):
    """Доля угаданных направлений среди ненулевых голосов и z против случайного угадывания
    с той же долей бычьих голосов (π — доля растущих периодов на проверке)."""
    m = vote != 0
    n = int(m.sum())
    if n < 20:
        return n, np.nan, np.nan
    call_up = vote[m] > 0
    hit = float(np.mean(call_up == up[m]))
    q = call_up.mean()
    expected = q * pi + (1 - q) * (1 - pi)
    z = (hit - expected) / np.sqrt(expected * (1 - expected) / n)
    return n, hit, z


def evaluate(days_list=(1, 7), refit_days=30, min_train_days=365):
    from sklearn.linear_model import LogisticRegression
    from config import SYMBOL, DB_DSN, TIMEFRAME, candles_per_day
    from db_layer import DatabaseLayer
    from vol_model import diebold_mariano

    cpd = candles_per_day(TIMEFRAME)
    df = DatabaseLayer(DB_DSN).get_all_data(SYMBOL, TIMEFRAME).sort_values('time').reset_index(drop=True)
    V = compute_votes(df, cpd)
    close = df['close'].values
    print(f"📦 {len(df)} свечей TF={TIMEFRAME}, голоса посчитаны")
    results = {}

    for days in days_list:
        H = days * cpd
        up_all = np.r_[close[H:] > close[:-H], np.full(H, np.nan)]   # рост через H свечей (NaN — будущее)
        first = int(np.argmax(V['warm'].values)) + min_train_days * cpd
        origins = np.arange(first, len(df) - H, H)                    # непересекающиеся периоды
        rows, model, fitted_at = [], None, -10 ** 9
        for t in origins:
            tr = np.arange(int(np.argmax(V['warm'].values)), t - H + 1)   # purge: цель известна к t
            y_tr = up_all[tr].astype(int)
            if t - fitted_at >= refit_days * cpd:
                model = LogisticRegression(C=1.0, max_iter=1000).fit(V.loc[tr, VOTE_COLS].values, y_tr)
                fitted_at = t
            rows.append({
                'up': bool(up_all[t]),
                'p_base': y_tr.mean(),
                'p_logit': model.predict_proba(V.loc[[t], VOTE_COLS].values)[0, 1],
                'p_equal': 0.5 + V.loc[t, 'vote_equal'] / 2,       # голос как «уверенность» старых ботов
                **{c: V.loc[t, c] for c in VOTE_COLS + ['vote_equal']},
            })
        R = pd.DataFrame(rows)
        up = R['up'].values
        pi = up.mean()

        print("\n" + "=" * 84)
        print(f"ГОРИЗОНТ {days} дн.: {len(R)} непересекающихся периодов, доля роста {pi:.1%}")
        print("=" * 84)
        print(f"  {'голос':14s} {'ненулевых':>9s} {'угадано':>8s} {'z (vs случай)':>14s}")
        for c in ['vote_equal'] + VOTE_COLS:
            n, hit, z = _hit_stats(R[c].values, up, pi)
            tag = 'ИТОГ (равные веса)' if c == 'vote_equal' else c
            print(f"  {tag:18s} {n:5d} {hit:8.1%} {z:+13.2f}" if n >= 20 else f"  {tag:18s} {n:5d}   мало данных")

        y = up.astype(float)
        bs = {k: (R[k].values - y) ** 2 for k in ('p_base', 'p_logit', 'p_equal')}
        print(f"\n  Brier (меньше — лучше): база {bs['p_base'].mean():.4f} | "
              f"авто-веса {bs['p_logit'].mean():.4f} ({(1 - bs['p_logit'].mean() / bs['p_base'].mean()) * 100:+.1f}%, "
              f"DM t={diebold_mariano(bs['p_logit'], bs['p_base']):+.2f}) | "
              f"голос как вероятность {bs['p_equal'].mean():.4f} ({(1 - bs['p_equal'].mean() / bs['p_base'].mean()) * 100:+.1f}%)")
        coef = dict(zip(VOTE_COLS, model.coef_[0].round(3)))
        print(f"  Авто-веса последней модели (логистическая регрессия): {coef}")
        results[days] = R
    return results


if __name__ == "__main__":
    evaluate()

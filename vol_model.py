# vol_model.py
# HAR-модель волатильности — основа коридора прогноза.
#
# Идея: направление недельного движения BTC практически непредсказуемо, а его масштаб
# (волатильность) — предсказуем. Поэтому:
#   1. HAR (Corsi, 2009) прогнозирует реализованную волатильность на горизонт:
#      log RV(след. 7 дней) = b0 + b1·log RV(день) + b2·log RV(неделя) + b3·log RV(месяц)
#   2. Квантили изменения цены = эмпирические квантили стандартизованной доходности z = ret / σ̂
#      (по обучающей истории) × σ̂ на момент прогноза.
# Коридор сужается в спокойные периоды и расширяется в бурные.
#
# Запуск обучения: python vol_model.py
import os

import joblib
import numpy as np
import pandas as pd

from config import SYMBOL, DB_DSN, TIMEFRAME, FORECAST_DAYS, MOVE_THRESHOLDS, candles_per_day

QUANTILES = [0.05, 0.5, 0.95]
MIN_TRAIN_DAYS = 60     # Минимум истории (после прогрева) для первого шага walk-forward
WF_STEP_DAYS = 10       # Шаг walk-forward проверки
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "har_model.pkl")
REPORT_PATH = os.path.join(MODEL_DIR, "walk_forward_har.csv")

HAR_COLS = ['har_d', 'har_w', 'har_m']
REGIMES = ('CALM', 'NORMAL', 'VOLATILE')
Z_GRID_LEVELS = np.linspace(0, 1, 201)   # распределение z храним 201 перцентилем
EPS = 1e-12


def har_frame(df, cpd, horizon):
    """HAR-признаки и цели для каждой свечи.

    Признаки строки t используют только свечи <= t.
    Цели (y_logrv, ret) — будущее: известны только через `horizon` свечей.
    """
    c = df['close'].values.astype(float)
    r = np.r_[np.nan, np.diff(np.log(c))]
    r2 = pd.Series(r ** 2)

    # RV, приведённая к "дневному" масштабу: сумма r² за окно / число дней в окне
    F = pd.DataFrame({
        'time': df['time'].values,
        'close': c,
        'har_d': np.log(r2.rolling(cpd).sum().clip(lower=EPS)),
        'har_w': np.log((r2.rolling(7 * cpd).sum() / 7).clip(lower=EPS)),
        'har_m': np.log((r2.rolling(30 * cpd).sum() / 30).clip(lower=EPS)),
    })
    future_rv = r2[::-1].rolling(horizon).sum()[::-1].shift(-horizon)   # Σ r² по свечам t+1..t+H
    F['y_logrv'] = np.log(future_rv.clip(lower=EPS))
    F['ret'] = (pd.Series(c).shift(-horizon) / c - 1) * 100
    F['warm'] = np.arange(len(F)) >= 30 * cpd    # признаки полностью прогреты
    return F


def _design(X):
    return np.c_[np.ones(len(X)), np.asarray(X, dtype=float)]


def fit_har(train):
    """OLS на log RV + квантили стандартизованной доходности. train — строки с известной целью."""
    X = _design(train[HAR_COLS])
    beta, *_ = np.linalg.lstsq(X, train['y_logrv'].values, rcond=None)
    resid = train['y_logrv'].values - X @ beta
    smear = float(np.mean(np.exp(resid)))     # поправка Даффа при переходе из логарифма в уровень
    sigma = np.sqrt(np.exp(X @ beta) * smear) * 100   # σ доходности за горизонт, в %
    z = train['ret'].values / sigma
    return {
        'beta': beta,
        'smear': smear,
        'z_quantiles': np.quantile(z, QUANTILES),
        'z_grid': np.quantile(z, Z_GRID_LEVELS),
        # Границы режимов: терцили исторического прогноза σ
        'sigma_terciles': np.quantile(sigma, [1 / 3, 2 / 3]),
        'sigma_sorted': np.sort(sigma)[::max(1, len(sigma) // 500)],
    }


def move_probability(params, sigma, threshold_pct):
    """P(|изменение цены| > threshold) по эмпирическому распределению z = ret / σ."""
    z_lo, z_hi = -threshold_pct / sigma, threshold_pct / sigma
    cdf = lambda x: float(np.interp(x, params['z_grid'], Z_GRID_LEVELS))
    return cdf(z_lo) + (1.0 - cdf(z_hi))


def classify_regime(params, sigma):
    """Режим волатильности и перцентиль σ среди исторических прогнозов."""
    lo, hi = params['sigma_terciles']
    regime = REGIMES[0] if sigma < lo else REGIMES[2] if sigma > hi else REGIMES[1]
    pct = float(np.searchsorted(params['sigma_sorted'], sigma) / len(params['sigma_sorted']) * 100)
    return regime, pct


def predict_sigma(params, har_row):
    x = np.r_[1.0, np.asarray(har_row, dtype=float)]
    return float(np.sqrt(np.exp(x @ params['beta']) * params['smear']) * 100)


def predict_quantiles(params, har_row):
    return params['z_quantiles'] * predict_sigma(params, har_row)


def pinball_loss(y, pred, quantiles=QUANTILES):
    q = np.asarray(quantiles)
    d = y - np.asarray(pred)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def diebold_mariano(loss_a, loss_b, lag=0):
    """t-статистика DM с поправкой Newey-West. < 0: A лучше B; |t| > 1.96 — значимо (5%)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    return float(d.mean() / np.sqrt(var / n)) if var > 0 else 0.0


def train_set(F, origin, horizon):
    """Строки, чья цель полностью известна к моменту `origin` (purge)."""
    tr = F.iloc[:origin - horizon + 1]
    return tr[tr['warm']].dropna(subset=HAR_COLS + ['y_logrv', 'ret'])


def walk_forward(F, cpd, horizon):
    first = 30 * cpd + MIN_TRAIN_DAYS * cpd
    step = WF_STEP_DAYS * cpd
    rows = []
    for o in range(first, len(F) - horizon, step):
        tr = train_set(F, o, horizon)
        assert tr.index.max() + horizon <= o, "утечка: цель обучающей строки заходит за точку прогноза"
        params = fit_har(tr)
        pred = predict_quantiles(params, F.loc[o, HAR_COLS])
        base = np.quantile(tr['ret'].values, QUANTILES)
        y = F.loc[o, 'ret']
        sig = predict_sigma(params, F.loc[o, HAR_COLS])
        regime, _ = classify_regime(params, sig)
        moves = {f'p_move{int(t)}': move_probability(params, sig, t) for t in MOVE_THRESHOLDS}
        base_moves = {f'base_move{int(t)}': float(np.mean(np.abs(tr['ret']) > t)) for t in MOVE_THRESHOLDS}
        rows.append({
            'regime': regime,
            **moves, **base_moves,
            'origin_time': F.loc[o, 'time'],
            'actual': y,
            'sigma': predict_sigma(params, F.loc[o, HAR_COLS]),
            'pred_q05': pred[0], 'pred_q50': pred[1], 'pred_q95': pred[2],
            'base_q05': base[0], 'base_q50': base[1], 'base_q95': base[2],
            'model_loss': pinball_loss(y, pred),
            'base_loss': pinball_loss(y, base),
            'in_interval': pred[0] <= y <= pred[2],
            'base_in_interval': base[0] <= y <= base[2],
        })
    return pd.DataFrame(rows)


def print_report(rep, horizon, step):
    n = len(rep)
    impr = (1 - rep['model_loss'].mean() / rep['base_loss'].mean()) * 100
    # Шаги проверки перекрываются по цели, если шаг меньше горизонта
    lag = int(np.ceil(horizon / step)) - 1
    dm = diebold_mariano(rep['model_loss'], rep['base_loss'], lag)
    width = rep['pred_q95'] - rep['pred_q05']
    base_width = rep['base_q95'] - rep['base_q05']

    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЁТ HAR (walk-forward с purge)")
    print("=" * 80)
    print(f"Шагов проверки: {n}")
    print(f"Pinball loss: HAR={rep['model_loss'].mean():.3f}, база={rep['base_loss'].mean():.3f} "
          f"({impr:+.1f}% к базе), HAR лучше на {np.mean(rep['model_loss'] < rep['base_loss']):.0%} шагов")
    print(f"Тест Диболда-Мариано: t={dm:+.2f} ({'значимо' if abs(dm) > 1.96 else 'не значимо'}; < 0 — HAR лучше)")
    print(f"Покрытие [q05, q95]: HAR {rep['in_interval'].mean():.0%}, база {rep['base_in_interval'].mean():.0%} (цель ~90%)")
    print(f"Средняя ширина коридора: HAR {width.mean():.1f} п.п., база {base_width.mean():.1f} п.п.")

    regime = pd.qcut(rep['sigma'], 3, labels=['спокойно', 'средне', 'бурно'])
    print("По режимам волатильности (покрытие / ширина):")
    for t in regime.cat.categories:
        m = regime == t
        print(f"  {t:9s} HAR {rep['in_interval'][m].mean():4.0%} / {width[m].mean():5.1f}   "
              f"база {rep['base_in_interval'][m].mean():4.0%} / {base_width[m].mean():5.1f}")

    # Проверка сигнала: предсказанный режим vs фактический размах движения
    absr = rep['actual'].abs()
    print("Сигнал по режиму (режим определён только по прошлым данным):")
    for reg_name in REGIMES:
        m = rep['regime'] == reg_name
        if m.any():
            cols = "  ".join(f">{int(t)}%: прогноз {rep[f'p_move{int(t)}'][m].mean():4.0%} / факт {np.mean(absr[m] > t):4.0%}"
                             for t in MOVE_THRESHOLDS)
            print(f"  {reg_name:9s} шагов {m.sum():3d}  ср.|движение| {absr[m].mean():4.1f}%  {cols}")
    for t in MOVE_THRESHOLDS:
        hit = (absr > t).astype(float)
        bs_model = np.mean((rep[f'p_move{int(t)}'] - hit) ** 2)
        bs_base = np.mean((rep[f'base_move{int(t)}'] - hit) ** 2)
        print(f"  Brier P(|движение|>{int(t)}%): модель {bs_model:.4f}, база {bs_base:.4f} "
              f"({(1 - bs_model / bs_base) * 100:+.1f}% к базе)")
    return {'improvement_pct': impr, 'dm_t': dm, 'coverage': float(rep['in_interval'].mean()), 'n_steps': n}


def train():
    from db_layer import DatabaseLayer

    cpd = candles_per_day(TIMEFRAME)
    horizon = FORECAST_DAYS * cpd
    df = DatabaseLayer(DB_DSN).get_all_data(SYMBOL, TIMEFRAME).sort_values('time').reset_index(drop=True)
    print(f"📦 {len(df)} свечей TF={TIMEFRAME}: {df['time'].min()} — {df['time'].max()}")

    F = har_frame(df, cpd, horizon)
    rep = walk_forward(F, cpd, horizon)
    os.makedirs(MODEL_DIR, exist_ok=True)
    rep.to_csv(REPORT_PATH, index=False)
    metrics = print_report(rep, horizon, WF_STEP_DAYS * cpd)

    # Финальная модель на всей истории с известной целью
    params = fit_har(train_set(F, len(F) - 1, horizon))
    joblib.dump({
        **params,
        'quantiles': QUANTILES,
        'timeframe': TIMEFRAME,
        'candles_per_day': cpd,
        'forecast_horizon': FORECAST_DAYS,
        'forecast_horizon_candles': horizon,
        'last_date': df['time'].iloc[-1],
        'metrics': metrics,
    }, MODEL_PATH)
    print(f"\n💾 HAR-модель сохранена: {MODEL_PATH}")
    print(f"   β = {np.round(params['beta'], 3)}, z-квантили = {np.round(params['z_quantiles'], 3)}")


class HarForecaster:
    """Прогноз коридора на FORECAST_DAYS дней. Формат результата совместим с SignalGenerator и визуализацией."""

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"HAR-модель не найдена: {model_path}. Запустите vol_model.py для обучения.")
        self.params = joblib.load(model_path)
        self.quantiles = self.params['quantiles']
        self.candles_per_day = self.params['candles_per_day']
        self.forecast_horizon = self.params['forecast_horizon']
        self.timeframe = self.params['timeframe']
        self.min_rows = 30 * self.candles_per_day + 1
        print(f"✅ HAR-модель загружена: ТФ={self.timeframe}, прогноз={self.forecast_horizon} дней")

    def predict(self, df):
        df = df.sort_values('time').reset_index(drop=True)
        if len(df) < self.min_rows:
            raise ValueError(f"Недостаточно данных: {len(df)} свечей, нужно минимум {self.min_rows}")

        F = har_frame(df, self.candles_per_day, self.params['forecast_horizon_candles'])
        last = F.iloc[-1]
        sigma = predict_sigma(self.params, last[HAR_COLS])
        changes = self.params['z_quantiles'] * sigma

        current_price = float(last['close'])
        regime, sigma_pct_rank = classify_regime(self.params, sigma)
        return {
            'regime': regime,
            'sigma_percentile': sigma_pct_rank,
            'move_probs': {t: move_probability(self.params, sigma, t) for t in MOVE_THRESHOLDS},
            'current_price': current_price,
            'current_date': last['time'],
            'forecast_horizon': self.forecast_horizon,
            'sigma_pct': sigma,
            'quantiles': {
                q: {'change_pct': float(ch), 'target_price': current_price * (1 + ch / 100)}
                for q, ch in zip(self.quantiles, changes)
            },
        }


if __name__ == "__main__":
    train()

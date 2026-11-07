# ensemble.py
# Смесь HAR (основа коридора) и LightGBM на индикаторах.
#
# Для каждого квантиля отдельно: q = w · LightGBM + (1 − w) · HAR.
# Вес w подбирается по walk-forward с purge. В отчёте вес на каждом шаге считается
# только по прошлым шагам (честная оценка); финальные веса — по всем шагам.
# Если LightGBM где-то не помогает, вес там ≈ 0 и прогноз совпадает с HAR.
#
# Запуск обучения (обучает HAR, LightGBM и подбирает веса): python ensemble.py
import os
import time

import joblib
import numpy as np
import pandas as pd

import model_trainer as mt
import vol_model as vm
from config import SYMBOL, DB_DSN, TIMEFRAME

WEIGHT_PATH = os.path.join(vm.MODEL_DIR, "ensemble_weights.pkl")
REPORT_PATH = os.path.join(vm.MODEL_DIR, "walk_forward_ensemble.csv")
WEIGHT_GRID = np.linspace(0, 1, 21)
MIN_PAST_STEPS = 40       # Сколько шагов нужно, прежде чем подбирать вес в честной оценке
QUANTILES = np.array(vm.QUANTILES)


def blend(w, lgbm, har):
    """Смешивание по квантилям + сортировка (квантили не должны пересекаться)."""
    return np.sort(np.asarray(w) * np.asarray(lgbm) + (1 - np.asarray(w)) * np.asarray(har), axis=-1)


def pinball_matrix(y, pred):
    """Потери по шагам и квантилям: shape (n, 3)."""
    d = np.asarray(y)[:, None] - np.asarray(pred)
    return np.maximum(QUANTILES * d, (QUANTILES - 1) * d)


def best_weights(y, lgbm, har):
    """Лучший вес LightGBM для каждого квантиля по сетке."""
    losses = np.array([pinball_matrix(y, w * lgbm + (1 - w) * har).mean(0) for w in WEIGHT_GRID])
    return WEIGHT_GRID[np.argmin(losses, axis=0)]


def aligned_walk_forward(raw):
    """HAR и LightGBM на одних и тех же точках прогноза, обе с purge."""
    cpd, H, W = mt.CPD, mt.FORECAST_HORIZON, mt.WINDOW_SIZE
    F = vm.har_frame(raw, cpd, H)
    win = mt.build_sliding_windows(raw.copy())          # окно k оканчивается на свече k + W − 1
    feat = [c for c in win.columns if c not in ['window_start', 'window_end', 'last_close', 'target']]

    first = 30 * cpd + vm.MIN_TRAIN_DAYS * cpd
    origins = list(range(first, len(F) - H, vm.WF_STEP_DAYS * cpd))
    rows, t0 = [], time.time()
    for i, o in enumerate(origins):
        k = o - W + 1
        assert win['window_end'].iloc[k] == F.loc[o, 'time'], "окно LightGBM и точка HAR не совпадают"
        y = F.loc[o, 'ret']

        tr_h = vm.train_set(F, o, H)
        assert tr_h.index.max() + H <= o, "утечка в HAR"
        params = vm.fit_har(tr_h)
        har = vm.predict_quantiles(params, F.loc[o, vm.HAR_COLS])

        train = win.iloc[:k - H]                         # purge: цели заканчиваются до точки прогноза
        assert train.index.max() + W - 1 + H <= o, "утечка в LightGBM"
        lgbm = mt.predict_quantiles(mt.fit_quantile_models(train[feat], train['target']),
                                    win.iloc[[k]][feat])[0]
        base = np.quantile(tr_h['ret'], QUANTILES)
        rows.append({'origin_time': F.loc[o, 'time'], 'actual': y,
                     'sigma': vm.predict_sigma(params, F.loc[o, vm.HAR_COLS]),
                     'har': har, 'lgbm': lgbm, 'base': base})
        if i % 50 == 0:
            print(f"  walk-forward шаг {i + 1}/{len(origins)} ({time.time() - t0:.0f} с)", flush=True)
    return rows


def evaluate(rows):
    y = np.array([r['actual'] for r in rows])
    P = {m: np.array([r[m] for r in rows]) for m in ('har', 'lgbm', 'base')}
    n = len(y)

    # Честная смесь: вес на шаге t подбирается только по шагам < t
    # (шаг walk-forward 60 свечей > горизонта 42, поэтому факт прошлых шагов уже известен)
    ens = np.full((n, 3), np.nan)
    weights_hist = np.full((n, 3), np.nan)
    for t in range(MIN_PAST_STEPS, n):
        w = best_weights(y[:t], P['lgbm'][:t], P['har'][:t])
        weights_hist[t] = w
        ens[t] = blend(w, P['lgbm'][t], P['har'][t])

    ev = np.arange(n) >= MIN_PAST_STEPS
    base_l = pinball_matrix(y[ev], P['base'][ev]).mean(1)
    har_l = pinball_matrix(y[ev], P['har'][ev]).mean(1)

    print("\n" + "=" * 80)
    print(f"ИТОГОВЫЙ ОТЧЁТ СМЕСИ HAR + LightGBM (честная оценка на {ev.sum()} шагах)")
    print("=" * 80)
    for name, pred in (('база', P['base']), ('HAR', P['har']), ('LightGBM', P['lgbm']), ('смесь', ens)):
        l = pinball_matrix(y[ev], pred[ev]).mean(1)
        cov = np.mean((y[ev] >= pred[ev, 0]) & (y[ev] <= pred[ev, 2]))
        print(f"  {name:9s} pinball {l.mean():.3f}  к базе {(1 - l.mean() / base_l.mean()) * 100:+5.1f}%  "
              f"DM t={vm.diebold_mariano(l, base_l):+5.2f}  покрытие {cov:4.0%}  "
              f"ширина {np.mean(pred[ev, 2] - pred[ev, 0]):5.1f}")
    ens_l = pinball_matrix(y[ev], ens[ev]).mean(1)
    dm = vm.diebold_mariano(ens_l, har_l)
    print(f"  Смесь против HAR: DM t={dm:+.2f} ({'значимо' if abs(dm) > 1.96 else 'не значимо'}; < 0 — смесь лучше)")
    print(f"  Средний вес LightGBM (q05/q50/q95): {np.nanmean(weights_hist, axis=0).round(2)}")

    final_w = best_weights(y, P['lgbm'], P['har'])
    report = pd.DataFrame({
        'origin_time': [r['origin_time'] for r in rows], 'actual': y,
        'sigma': [r['sigma'] for r in rows],
        **{f'{m}_q{int(q * 100):02d}': P[m][:, j] for m in ('har', 'lgbm', 'base') for j, q in enumerate(QUANTILES)},
        **{f'ens_q{int(q * 100):02d}': ens[:, j] for j, q in enumerate(QUANTILES)},
        **{f'w_q{int(q * 100):02d}': weights_hist[:, j] for j, q in enumerate(QUANTILES)},
    })
    metrics = {
        'ens_vs_base_pct': float((1 - ens_l.mean() / base_l.mean()) * 100),
        'ens_vs_har_pct': float((1 - ens_l.mean() / har_l.mean()) * 100),
        'dm_ens_vs_har': dm,
        'coverage': float(np.mean((y[ev] >= ens[ev, 0]) & (y[ev] <= ens[ev, 2]))),
        'n_steps': int(ev.sum()),
    }
    return final_w, report, metrics


def train():
    from db_layer import DatabaseLayer

    raw = DatabaseLayer(DB_DSN).get_all_data(SYMBOL, TIMEFRAME).sort_values('time').reset_index(drop=True)
    print(f"📦 {len(raw)} свечей TF={TIMEFRAME}: {raw['time'].min()} — {raw['time'].max()}")

    # 1. Финальные модели (каждый тренер печатает свой отчёт и сохраняет .pkl)
    print("\n### HAR")
    vm.train()
    print("\n### LightGBM")
    mt.train_walk_forward()

    # 2. Совместный walk-forward и подбор весов
    print("\n### Подбор весов смеси")
    rows = aligned_walk_forward(raw)
    final_w, report, metrics = evaluate(rows)
    report.to_csv(REPORT_PATH, index=False)
    joblib.dump({'weights': final_w, 'quantiles': list(QUANTILES), 'metrics': metrics,
                 'last_date': raw['time'].iloc[-1]}, WEIGHT_PATH)
    print(f"\n💾 Веса смеси (вес LightGBM для q05/q50/q95): {final_w} → {WEIGHT_PATH}")


class EnsembleForecaster:
    """HAR + LightGBM. Без файла весов или модели LightGBM — чистый HAR."""

    def __init__(self, har_path=vm.MODEL_PATH, weight_path=WEIGHT_PATH):
        self.har = vm.HarForecaster(har_path)
        self.candles_per_day = self.har.candles_per_day
        self.forecast_horizon = self.har.forecast_horizon
        self.lgbm, self.weights = None, None
        try:
            from forecaster import Forecaster
            self.weights = np.asarray(joblib.load(weight_path)['weights'])
            self.lgbm = Forecaster()
            self.lgbm.conformal_correction = 0.0   # веса подобраны на сырых квантилях LightGBM
            print(f"✅ Смесь включена: вес LightGBM (q05/q50/q95) = {self.weights}")
        except FileNotFoundError as e:
            print(f"⚠️  {e}\n   Прогноз только по HAR. Для смеси запустите: python ensemble.py")
            self.lgbm, self.weights = None, None

    def predict(self, df):
        fc = self.har.predict(df)
        if self.lgbm is None:
            fc['model'] = 'HAR'
            return fc

        lg = self.lgbm.predict(df)
        assert pd.Timestamp(lg['current_date']) == pd.Timestamp(fc['current_date']), "модели смотрят на разные свечи"
        qs = list(fc['quantiles'])
        har_pct = np.array([fc['quantiles'][q]['change_pct'] for q in qs])
        lg_pct = np.array([lg['quantiles'][q]['change_pct'] for q in qs])
        mixed = blend(self.weights, lg_pct, har_pct)

        price = fc['current_price']
        fc['quantiles'] = {q: {'change_pct': float(ch), 'target_price': price * (1 + ch / 100)}
                           for q, ch in zip(qs, mixed)}
        fc['model'] = 'HAR + LightGBM'
        return fc


if __name__ == "__main__":
    train()

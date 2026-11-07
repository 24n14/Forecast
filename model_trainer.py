# model_trainer_v2.py
# Скользящее окно: один пример = 30 дневных свечей → прогноз на следующие 7 дней
import os
import warnings
warnings.filterwarnings('ignore')

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import SYMBOL, DB_DSN
from ta_engine import add_features, aggregate_window

# Параметры
WINDOW_SIZE = 30           # Размер окна (дней) для одного примера
FORECAST_HORIZON = 7       # Прогноз на N дней вперёд (1 неделя)
MIN_TRAIN_WINDOWS = 60     # Минимум окон для первого обучения (~2 месяца)
STEP = 1                   # Шаг скольжения (1 = максимальное перекрытие)
MODEL_DIR = "models"
QUANTILES = [0.05, 0.5, 0.95]

# LightGBM: простая модель для защиты от переобучения
LGB_PARAMS = dict(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=15,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
    verbose=-1,
)


def build_sliding_windows(df):
    """Строит датасет скользящих окон."""
    df = df.sort_values('time').reset_index(drop=True)

    # Применяем индикаторы один раз на всей истории
    df = add_features(df)

    rows = []
    # Окно должно вмещаться полностью + цель должна быть известна
    for i in range(WINDOW_SIZE, len(df) - FORECAST_HORIZON + 1, STEP):
        window_df = df.iloc[i - WINDOW_SIZE:i].copy()

        # Агрегируем окно
        features = aggregate_window(window_df)
        features['window_start'] = window_df['time'].iloc[0]
        features['window_end'] = window_df['time'].iloc[-1]
        features['last_close'] = window_df['close'].iloc[-1]

        # Цель: изменение цены через FORECAST_HORIZON свечей
        future_close = df['close'].iloc[i + FORECAST_HORIZON - 1]
        features['target'] = (future_close / features['last_close'] - 1) * 100

        rows.append(features)

    return pd.DataFrame(rows)


def fit_quantile_models(X, y, quantiles=QUANTILES):
    models = {}
    for q in quantiles:
        m = lgb.LGBMRegressor(objective='quantile', alpha=q, **LGB_PARAMS)
        m.fit(X, y, callbacks=[lgb.log_evaluation(period=0)])
        models[q] = m
    return models


def predict_quantiles(models, X):
    preds = np.column_stack([models[q].predict(X) for q in sorted(models)])
    return np.sort(preds, axis=1)


def pinball_loss(y, pred, q):
    diff = y - pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def train_walk_forward():
    """Walk-forward валидация по скользящим окнам."""
    from db_layer import DatabaseLayer

    db = DatabaseLayer(DB_DSN)
    df_raw = db.get_all_data(SYMBOL)

    print(f"📦 Загружено {len(df_raw)} свечей: {df_raw['time'].min().date()} — {df_raw['time'].max().date()}")

    # Строим датасет окон
    df = build_sliding_windows(df_raw)
    feature_cols = [col for col in df.columns if col not in ['window_start', 'window_end', 'last_close', 'target']]
    n_windows = len(df)

    print(f"📊 Скользящих окон: {n_windows}, размер окна: {WINDOW_SIZE} дней, прогноз: {FORECAST_HORIZON} дней")
    print(f"Признаков: {len(feature_cols)}")
    print(f"Первое окно: {df['window_start'].iloc[0].date()} — {df['window_end'].iloc[0].date()}")
    print(f"Последнее окно: {df['window_start'].iloc[-1].date()} — {df['window_end'].iloc[-1].date()}\n")

    if n_windows < MIN_TRAIN_WINDOWS + 10:
        raise ValueError(f"Недостаточно окон: {n_windows}, нужно минимум {MIN_TRAIN_WINDOWS + 10}")

    # Walk-forward: учимся на первых k окнах, проверяем на k+1
    # Проверяем каждые 10 окон (~10 дней)
    results = []
    test_steps = list(range(MIN_TRAIN_WINDOWS, n_windows, 10))

    for k in test_steps:
        train = df.iloc[:k]
        test = df.iloc[[k]]

        X_train, y_train = train[feature_cols], train['target']
        X_test, y_test = test[feature_cols], test['target']

        # Обучаем
        models = fit_quantile_models(X_train, y_train)
        pred = predict_quantiles(models, X_test)[0]

        # Baseline
        baseline = np.quantile(y_train, QUANTILES)

        y_val = y_test.iloc[0]
        model_loss = np.mean([pinball_loss(np.array([y_val]), np.array([pred[i]]), q)
                              for i, q in enumerate(QUANTILES)])
        base_loss = np.mean([pinball_loss(np.array([y_val]), np.array([baseline[i]]), q)
                             for i, q in enumerate(QUANTILES)])

        in_interval = (y_val >= pred[0]) and (y_val <= pred[-1])
        median_error = abs(y_val - pred[1])

        result = {
            'step': len(results) + 1,
            'train_windows': k,
            'test_window_end': test['window_end'].iloc[0],
            'model_loss': model_loss,
            'base_loss': base_loss,
            'in_interval': in_interval,
            'median_error': median_error,
            'actual': y_val,
            'pred_q05': pred[0],
            'pred_q50': pred[1],
            'pred_q95': pred[2],
        }
        results.append(result)

        status = "✓" if in_interval else "✗"
        better = "✓" if model_loss < base_loss else "✗"
        print(f"{status} Шаг {result['step']:2d} | Обучение: {k:3d} окон | "
              f"Проверка до: {test['window_end'].iloc[0].date()} | "
              f"Факт: {y_val:+6.2f}% | "
              f"Прогноз [{pred[0]:+6.2f}, {pred[1]:+6.2f}, {pred[2]:+6.2f}] | "
              f"{better} Loss: модель={model_loss:.3f} база={base_loss:.3f}")

    # Итоговый отчёт
    report = pd.DataFrame(results)
    os.makedirs(MODEL_DIR, exist_ok=True)
    report.to_csv(os.path.join(MODEL_DIR, 'walk_forward_sliding.csv'), index=False)

    coverage = report['in_interval'].mean()
    wins = (report['model_loss'] < report['base_loss']).mean()

    print("\n" + "="*90)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("="*90)
    print(f"Шагов проверки: {len(report)}")
    print(f"Покрытие интервала [q05, q95]: {coverage:.0%} (цель ~90%)")
    print(f"Модель лучше базы: {wins:.0%} шагов")
    print(f"Средняя pinball loss: модель={report['model_loss'].mean():.3f}, "
          f"база={report['base_loss'].mean():.3f}")
    print(f"Средняя ошибка медианы: {report['median_error'].mean():.2f} п.п.")

    improvement = (1 - report['model_loss'].mean() / report['base_loss'].mean()) * 100
    print(f"\n{'✅' if improvement > 0 else '❌'} Улучшение над базой: {improvement:+.1f}%\n")

    # Финальная модель на всех окнах
    X_full = df[feature_cols]
    y_full = df['target']
    final_models = fit_quantile_models(X_full, y_full)

    model_path = os.path.join(MODEL_DIR, 'sliding_quantile_model.pkl')
    joblib.dump({
        'models': final_models,
        'features': feature_cols,
        'quantiles': QUANTILES,
        'window_size': WINDOW_SIZE,
        'forecast_horizon': FORECAST_HORIZON,
        'trained_windows': n_windows,
        'last_date': df['window_end'].iloc[-1],
    }, model_path)

    print(f"💾 Финальная модель ({n_windows} окон) сохранена: {model_path}")

    # Прогноз на текущее окно
    current_window = df.iloc[[-1]]
    X_current = current_window[feature_cols]
    pred_current = predict_quantiles(final_models, X_current)[0]
    current_close = current_window['last_close'].iloc[0]

    print(f"\n🔮 ПРОГНОЗ на следующие {FORECAST_HORIZON} дней")
    print(f"От: {current_window['window_end'].iloc[0].date()} (close ${current_close:,.1f})")
    print(f"Квантили изменения цены:")
    for q, v in zip(QUANTILES, pred_current):
        target_price = current_close * (1 + v / 100)
        print(f"   q{int(q*100):02d}: {v:+6.2f}% → ${target_price:,.1f}")

    return report


if __name__ == "__main__":
    train_walk_forward()

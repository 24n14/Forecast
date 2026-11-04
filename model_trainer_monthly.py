# model_trainer.py
# Месячная агрегация: один пример = один месяц, прогноз на следующий месяц
import os
import warnings
warnings.filterwarnings('ignore')

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import SYMBOL, DB_DSN
from ta_engine import add_features

# Параметры обучения
MIN_TRAIN_MONTHS = 3         # Минимум месяцев для первого обучения
MODEL_DIR = "models"
QUANTILES = [0.05, 0.5, 0.95]

# LightGBM с усиленной защитой от переобучения
LGB_PARAMS = dict(
    n_estimators=50,         # Ещё меньше деревьев
    learning_rate=0.1,       # Более агрессивное обучение
    max_depth=2,             # Очень мелкие деревья
    num_leaves=4,            # Минимум листьев
    min_child_samples=3,     # Минимум примеров в листе
    subsample=0.6,           # Больше шума
    subsample_freq=1,
    colsample_bytree=0.6,    # Больше шума
    reg_alpha=1.0,           # Сильная L1 регуляризация
    reg_lambda=5.0,          # Очень сильная L2 регуляризация
    verbose=-1,
)


def aggregate_month(month_df):
    """Агрегирует свечи одного месяца в один пример с признаками."""
    # Проверяем минимальное количество свечей для индикаторов
    if len(month_df) < 25:
        return None

    # Применяем индикаторы на дневных свечах
    month_df = add_features(month_df.sort_values('time').reset_index(drop=True))

    c = month_df['close'].values
    first_close = c[0]
    last_close = c[-1]

    # Статистики цены
    features = {
        'month_return': (last_close / first_close - 1) * 100,
        'volatility': np.std(np.diff(np.log(c))) * np.sqrt(len(c)) * 100,
        'max_drawdown': (c.min() / c.max() - 1) * 100,
        'price_range_pct': (c.max() / c.min() - 1) * 100,

        # Объём
        'avg_volume': month_df['volume'].mean(),
        'volume_trend': np.corrcoef(np.arange(len(month_df)), month_df['volume'])[0, 1] if len(month_df) > 1 else 0,
        'volume_std': month_df['volume'].std(),

        # Индикаторы
        'avg_rsi': month_df['rsi_14'].mean(),
        'rsi_min': month_df['rsi_14'].min(),
        'rsi_max': month_df['rsi_14'].max(),
        'close_to_sma20_mean': (month_df['close'] / month_df['sma_20'] - 1).mean() * 100,
        'close_to_sma20_last': (last_close / month_df['sma_20'].iloc[-1] - 1) * 100,

        # Паттерны
        'bullish_pattern_pct': month_df['is_bullish_pattern'].mean() * 100,

        # Тренд
        'price_trend': np.corrcoef(np.arange(len(c)), c)[0, 1] if len(c) > 1 else 0,

        # Динамика последней недели (7 свечей)
        'last_week_return': (c[-1] / c[-7] - 1) * 100 if len(c) >= 7 else 0,
        'last_week_vol': np.std(np.diff(np.log(c[-7:]))) * np.sqrt(7) * 100 if len(c) >= 7 else 0,
    }

    return features


def build_monthly_dataset(df):
    """Строит датасет: один пример = один месяц, цель = изменение до следующего."""
    df = df.sort_values('time').reset_index(drop=True)
    df['month'] = pd.to_datetime(df['time']).dt.to_period('M')

    months = df['month'].unique()
    if len(months) < MIN_TRAIN_MONTHS + 1:
        raise ValueError(f"Недостаточно данных: {len(months)} месяцев, нужно минимум {MIN_TRAIN_MONTHS + 1}")

    rows = []
    for i, month in enumerate(months):
        month_df = df[df['month'] == month].copy()

        # Агрегируем текущий месяц
        features = aggregate_month(month_df)
        if features is None:  # Пропускаем месяцы с недостаточным количеством свечей
            continue

        features['month'] = str(month)
        features['last_close'] = month_df['close'].iloc[-1]

        # Цель: изменение до закрытия следующего месяца (если есть)
        if i < len(months) - 1:
            next_month_df = df[df['month'] == months[i + 1]]
            next_close = next_month_df['close'].iloc[-1]
            features['target'] = (next_close / features['last_close'] - 1) * 100
        else:
            features['target'] = np.nan

        rows.append(features)

    return pd.DataFrame(rows)


def fit_quantile_models(X, y, quantiles=QUANTILES):
    """Обучает модели для каждого квантиля."""
    models = {}
    for q in quantiles:
        m = lgb.LGBMRegressor(objective='quantile', alpha=q, **LGB_PARAMS)
        m.fit(X, y, callbacks=[lgb.log_evaluation(period=0)])
        models[q] = m
    return models


def predict_quantiles(models, X):
    """Прогноз по всем квантилям с упорядочиванием."""
    preds = np.column_stack([models[q].predict(X) for q in sorted(models)])
    return np.sort(preds, axis=1)


def pinball_loss(y, pred, q):
    """Quantile loss (pinball loss)."""
    diff = y - pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def train_walk_forward():
    """Walk-forward валидация по месяцам."""
    from db_layer import DatabaseLayer

    db = DatabaseLayer(DB_DSN)
    df_raw = db.get_all_data(SYMBOL)

    print(f"📦 Загружено {len(df_raw)} свечей: {df_raw['time'].min().date()} — {df_raw['time'].max().date()}")

    # Строим месячный датасет
    df = build_monthly_dataset(df_raw)
    feature_cols = [col for col in df.columns if col not in ['month', 'last_close', 'target']]

    # Убираем последний месяц (у него нет цели)
    df = df[df['target'].notna()].reset_index(drop=True)
    n_months = len(df)

    print(f"📊 Месячных примеров: {n_months}, признаков: {len(feature_cols)}")
    print(f"Месяцы: {df['month'].iloc[0]} — {df['month'].iloc[-1]}")
    print(f"\nПризнаки: {', '.join(feature_cols)}\n")

    if n_months < MIN_TRAIN_MONTHS + 1:
        raise ValueError(f"Недостаточно месяцев: {n_months}, нужно минимум {MIN_TRAIN_MONTHS + 1}")

    # Walk-forward: учимся на [0..k], проверяем на [k+1]
    results = []
    for k in range(MIN_TRAIN_MONTHS, n_months):
        train = df.iloc[:k]
        test = df.iloc[[k]]

        X_train, y_train = train[feature_cols], train['target']
        X_test, y_test = test[feature_cols], test['target']

        # Обучаем модели
        models = fit_quantile_models(X_train, y_train)
        pred = predict_quantiles(models, X_test)[0]

        # Baseline: исторические квантили тренировочной выборки
        baseline = np.quantile(y_train, QUANTILES)

        # Метрики
        y_val = y_test.iloc[0]
        model_loss = np.mean([pinball_loss(np.array([y_val]), np.array([pred[i]]), q)
                              for i, q in enumerate(QUANTILES)])
        base_loss = np.mean([pinball_loss(np.array([y_val]), np.array([baseline[i]]), q)
                             for i, q in enumerate(QUANTILES)])

        in_interval = (y_val >= pred[0]) and (y_val <= pred[-1])
        median_error = abs(y_val - pred[1])

        result = {
            'step': k - MIN_TRAIN_MONTHS + 1,
            'train_months': k,
            'test_month': test['month'].iloc[0],
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
        print(f"{status} Шаг {result['step']:2d} | Обучение: {k:2d} месяцев | "
              f"Проверка: {test['month'].iloc[0]} | "
              f"Факт: {y_val:+6.2f}% | "
              f"Прогноз [{pred[0]:+6.2f}, {pred[1]:+6.2f}, {pred[2]:+6.2f}] | "
              f"Loss: модель={model_loss:.3f} база={base_loss:.3f}")

    # Итоговый отчёт
    report = pd.DataFrame(results)
    os.makedirs(MODEL_DIR, exist_ok=True)
    report.to_csv(os.path.join(MODEL_DIR, 'walk_forward_monthly.csv'), index=False)

    coverage = report['in_interval'].mean()
    wins = (report['model_loss'] < report['base_loss']).mean()

    print("\n" + "="*80)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("="*80)
    print(f"Шагов проверки: {len(report)}")
    print(f"Покрытие интервала [q05, q95]: {coverage:.0%} (цель ~90%)")
    print(f"Модель лучше базы: {wins:.0%} шагов")
    print(f"Средняя pinball loss: модель={report['model_loss'].mean():.3f}, "
          f"база={report['base_loss'].mean():.3f}")
    print(f"Средняя ошибка медианы: {report['median_error'].mean():.2f} п.п.\n")

    # Обучаем финальную модель на всех доступных месяцах
    X_full = df[feature_cols]
    y_full = df['target']
    final_models = fit_quantile_models(X_full, y_full)

    model_path = os.path.join(MODEL_DIR, 'monthly_quantile_model.pkl')
    joblib.dump({
        'models': final_models,
        'features': feature_cols,
        'quantiles': QUANTILES,
        'trained_months': n_months,
        'last_month': df['month'].iloc[-1],
    }, model_path)

    print(f"💾 Финальная модель ({n_months} месяцев) сохранена: {model_path}")

    # Прогноз на текущий (неполный) месяц
    current_month_df = df_raw[pd.to_datetime(df_raw['time']).dt.to_period('M') ==
                               pd.to_datetime(df_raw['time']).iloc[-1].to_period('M')]

    if len(current_month_df) >= 5:  # Хотя бы 5 свечей для индикаторов
        current_features = aggregate_month(current_month_df)
        current_close = current_month_df['close'].iloc[-1]
        X_current = pd.DataFrame([{k: current_features[k] for k in feature_cols}])

        pred_current = predict_quantiles(final_models, X_current)[0]

        print(f"\n🔮 ПРОГНОЗ на следующий месяц")
        print(f"От: {current_month_df['time'].iloc[-1].date()} (close {current_close:.1f})")
        print(f"Квантили изменения цены:")
        for q, v in zip(QUANTILES, pred_current):
            target_price = current_close * (1 + v / 100)
            print(f"   q{int(q*100):02d}: {v:+6.2f}% → ${target_price:,.1f}")

    return report


if __name__ == "__main__":
    train_walk_forward()

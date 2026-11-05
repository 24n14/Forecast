# forecaster.py
# Модуль для прогнозирования с использованием обученной модели
import os
import joblib
import numpy as np
import pandas as pd
from ta_engine import add_features

class Forecaster:
    """Загружает обученную модель и делает прогнозы на основе последних данных."""

    def __init__(self, model_path="models/sliding_quantile_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}. Запустите model_trainer.py для обучения.")

        # Загружаем модель и её параметры
        model_data = joblib.load(model_path)
        self.models = model_data['models']
        self.features = model_data['features']
        self.quantiles = model_data['quantiles']
        self.window_size = model_data['window_size']
        self.forecast_horizon = model_data['forecast_horizon']

        print(f"✅ Модель загружена: окно={self.window_size} дней, прогноз={self.forecast_horizon} дней")

    def aggregate_window(self, window_df):
        """Агрегирует окно свечей в признаки (идентично model_trainer.py)."""
        c = window_df['close'].values
        first_close = c[0]
        last_close = c[-1]

        features = {
            # Динамика цены
            'return': (last_close / first_close - 1) * 100,
            'volatility': np.std(np.diff(np.log(c))) * np.sqrt(len(c)) * 100,
            'max_drawdown': (c.min() / c.max() - 1) * 100,
            'price_range_pct': (c.max() / c.min() - 1) * 100,

            # Объём
            'avg_volume': window_df['volume'].mean(),
            'volume_std': window_df['volume'].std(),
            'volume_trend': np.corrcoef(np.arange(len(window_df)), window_df['volume'])[0, 1] if len(window_df) > 1 else 0,

            # Индикаторы
            'avg_rsi': window_df['rsi_14'].mean(),
            'rsi_min': window_df['rsi_14'].min(),
            'rsi_max': window_df['rsi_14'].max(),
            'rsi_last': window_df['rsi_14'].iloc[-1],
            'close_to_sma20_mean': (window_df['close'] / window_df['sma_20'] - 1).mean() * 100,
            'close_to_sma20_last': (last_close / window_df['sma_20'].iloc[-1] - 1) * 100,

            # Паттерны
            'bullish_pattern_count': window_df['is_bullish_pattern'].sum(),
            'bullish_pattern_pct': window_df['is_bullish_pattern'].mean() * 100,

            # Тренд
            'price_trend': np.corrcoef(np.arange(len(c)), c)[0, 1] if len(c) > 1 else 0,

            # Последняя неделя
            'last_week_return': (c[-1] / c[-7] - 1) * 100 if len(c) >= 7 else 0,
            'last_week_vol': np.std(np.diff(np.log(c[-7:]))) * np.sqrt(7) * 100 if len(c) >= 7 else 0,

            # Последние 3 дня
            'last_3d_return': (c[-1] / c[-3] - 1) * 100 if len(c) >= 3 else 0,
        }

        return features

    def predict(self, df):
        """
        Делает прогноз на основе последних window_size свечей.

        Args:
            df: DataFrame со свечами (должен быть отсортирован по времени)

        Returns:
            dict с прогнозом: {
                'current_price': float,
                'current_date': datetime,
                'forecast_horizon': int,
                'quantiles': {0.05: {...}, 0.5: {...}, 0.95: {...}}
            }
        """
        df = df.sort_values('time').reset_index(drop=True)

        if len(df) < self.window_size:
            raise ValueError(f"Недостаточно данных: {len(df)} свечей, нужно минимум {self.window_size}")

        # Применяем индикаторы
        df = add_features(df)

        # Берём последние window_size свечей
        window_df = df.iloc[-self.window_size:].copy()

        # Агрегируем признаки
        features = self.aggregate_window(window_df)
        X = pd.DataFrame([{k: features[k] for k in self.features}])

        # Прогноз по квантилям
        predictions = {}
        for q in self.quantiles:
            pred_change = self.models[q].predict(X)[0]
            predictions[q] = pred_change

        # Упорядочиваем квантили (на случай пересечения)
        sorted_values = np.sort(list(predictions.values()))
        predictions = {q: v for q, v in zip(self.quantiles, sorted_values)}

        # Текущая цена и дата
        current_price = window_df['close'].iloc[-1]
        current_date = window_df['time'].iloc[-1]

        # Формируем результат
        result = {
            'current_price': current_price,
            'current_date': current_date,
            'forecast_horizon': self.forecast_horizon,
            'quantiles': {}
        }

        for q, change_pct in predictions.items():
            target_price = current_price * (1 + change_pct / 100)
            result['quantiles'][q] = {
                'change_pct': change_pct,
                'target_price': target_price,
            }

        return result

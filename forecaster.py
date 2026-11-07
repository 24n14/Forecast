# forecaster.py
# Модуль для прогнозирования с использованием обученной модели
import os
import joblib
import numpy as np
import pandas as pd
from ta_engine import add_features, aggregate_window

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
        features = aggregate_window(window_df)
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

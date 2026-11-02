#import os
#from dotenv import load_dotenv

#load_dotenv()  # Обязательно: читает .env, чтобы взять DB_DSN

# --- Торговые настройки (то, что реально влияет на логику) ---
SYMBOL = "BTCUSDT"          # Пара для анализа
TIMEFRAME = "D"             # Таймфрейм: D, 60, 15 и т.д.
HISTORY_DAYS = 180          # Сколько дней истории брать из БД

# --- Параметры индикаторов и стратегий ---
QUANTILES = [0.05, 0.5, 0.95]       # Для прогноза (если будешь делать ML)
FORECAST_HORIZON = 22              # На сколько баров вперёд прогнозируем
INVALIDATION_PERCENT = 6.0         # Порог отмены сигнала (в %)

# --- База данных ---
DB_DSN = "sqlite:///test_bot.db"


"""
DB_DSN = os.getenv("DB_DSN")
if not DB_DSN:
    raise ValueError("❌ Ошибка: DB_DSN не найден в .env! Создай файл .env и пропиши туда строку подключения.")
    
# --- Флаги поведения (только то, что нужно) ---
# Если позже захочешь включать приватные функции (ордера, баланс), поставь true в .env
USE_PRIVATE_FEATURES = os.getenv("USE_PRIVATE_FEATURES", "False").lower() == "true"
"""

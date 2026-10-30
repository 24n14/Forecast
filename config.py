import os
from dotenv import load_dotenv

load_dotenv()

# Символ и таймфрейм
SYMBOL = "BTCUSDT"
TIMEFRAME = "D"

# Буфер истории (в днях)
HISTORY_DAYS = 180

# Параметры ML
QUANTILES = [0.05, 0.5, 0.95]  # Оставили только края и медиану для скорости
FORECAST_HORIZON = 22

# Порог отмены (Invalidation Threshold)
INVALIDATION_PERCENT = 6.0

# Конфигурация БД и API
DB_DSN = os.getenv("DB_DSN")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET = os.getenv("BYBIT_SECRET")
# Флаг теперь нужен только для проверки сделок, а не для свечей
USE_PRIVATE_FEATURES = os.getenv("USE_PRIVATE_FEATURES", "False").lower() == "true"

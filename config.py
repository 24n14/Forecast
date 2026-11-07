
# --- Торговые настройки (то, что реально влияет на логику) ---
SYMBOL = "BTCUSDT"          # Пара для анализа
TIMEFRAME = "240"           # Таймфрейм Bybit: "D" (день), "240" (4 часа), "60" (1 час)

# Длительность свечи в часах для каждого таймфрейма
TIMEFRAME_HOURS = {"D": 24, "240": 4, "120": 2, "60": 1}


def candles_per_day(timeframe=TIMEFRAME):
    return 24 // TIMEFRAME_HOURS[timeframe]


# --- Модель (в днях; в свечи пересчитывается по таймфрейму) ---
WINDOW_DAYS = 30            # История для одного примера
FORECAST_DAYS = 7           # Горизонт прогноза
HISTORY_START = "2020-04-01"  # С какой даты грузить историю для обучения

# --- Уведомления ---
NTFY_TOPIC = "forecaster_24n14"  # Топик ntfy.sh для уведомлений
ENABLE_NOTIFICATIONS = True      # Включить/выключить отправку уведомлений

# --- Параметры сигналов ---
SIGNAL_THRESHOLD_STRONG = 5.0    # Порог для сильного сигнала (в %, горизонт 7 дней)
SIGNAL_THRESHOLD_WEAK = 2.5      # Порог для слабого сигнала (в %, горизонт 7 дней)

# --- Визуализация ---
ENABLE_VISUALIZATION = True      # Создавать графики прогнозов
VIZ_HISTORY_DAYS = 60           # Сколько дней истории показывать на графике
VIZ_SAMPLES = 300               # Количество траекторий в точечном облаке

# --- Планировщик ---
SCHEDULE_ENABLED = False         # True = работа по расписанию, False = однократный запуск
SCHEDULE_TIME = "09:00"          # Время запуска в формате "HH:MM" (UTC)
SCHEDULE_TIMEZONE = "UTC"        # Временная зона

# --- База данных ---
DB_DSN = "sqlite:///test_bot.db"

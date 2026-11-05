
# --- Торговые настройки (то, что реально влияет на логику) ---
SYMBOL = "BTCUSDT"          # Пара для анализа
TIMEFRAME = "D"             # Таймфрейм: D, 60, 15 и т.д.

# --- Уведомления ---
NTFY_TOPIC = "forecaster_24n14"  # Топик ntfy.sh для уведомлений
ENABLE_NOTIFICATIONS = True      # Включить/выключить отправку уведомлений

# --- Параметры сигналов ---
SIGNAL_THRESHOLD_STRONG = 10.0   # Порог для сильного сигнала (в %)
SIGNAL_THRESHOLD_WEAK = 5.0      # Порог для слабого сигнала (в %)

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

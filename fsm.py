# fsm.py
from enum import Enum, auto
from forecaster import Forecaster
from signal_generator import SignalGenerator
from notifier import Notifier


class State(Enum):
    START = auto()
    DOWNLOAD = auto()
    FORECAST = auto()
    NOTIFY = auto()
    DONE = auto()
    ERROR = auto()


class TradingFSM:
    # ❌ БЫЛО: def __init__(self, client, db, symbol, config):
    # ✅ СТАЛО: передаём конкретные значения, а не весь конфиг
    def __init__(self, client, db, symbol, timeframe, history_days,
                 ntfy_topic=None, enable_notifications=True,
                 signal_threshold_strong=10.0, signal_threshold_weak=5.0):
        self.client = client
        self.db = db
        self.symbol = symbol
        self.timeframe = timeframe
        self.history_days = history_days

        # Загружаем модель прогнозирования
        try:
            self.forecaster = Forecaster()
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print("   Бот продолжит работу без прогнозов.")
            self.forecaster = None

        # Генератор сигналов
        self.signal_generator = SignalGenerator(
            threshold_strong=signal_threshold_strong,
            threshold_weak=signal_threshold_weak
        )

        # Уведомления
        self.enable_notifications = enable_notifications and ntfy_topic is not None
        if self.enable_notifications:
            self.notifier = Notifier(ntfy_topic)
            print(f"📢 Уведомления включены: {ntfy_topic}")
        else:
            self.notifier = None
            if not ntfy_topic:
                print("⚠️  Топик ntfy.sh не указан, уведомления отключены")

        self.state = State.START
        self.forecast = None
        self.signal = None

    def run(self):
        while self.state != State.DONE and self.state != State.ERROR:
            if self.state == State.START:
                self.state = State.DOWNLOAD

            elif self.state == State.DOWNLOAD:
                try:
                    print("📥 Загрузка свечей...")
                    # ✅ Используем self.timeframe вместо self.config.TIMEFRAME
                    raw_df = self.client.fetch_candles(
                        symbol=self.symbol,
                        interval=self.timeframe,
                        limit=1000
                    )

                    if raw_df is not None and not raw_df.empty:
                        self.db.save_candles(raw_df)
                        print("✅ Свечи сохранены в БД.")
                        self.state = State.FORECAST
                    else:
                        raise ValueError("Получен пустой DataFrame. Проверьте символ.")

                except Exception as e:
                    print(f"❌ Ошибка загрузки: {e}")
                    self.state = State.ERROR

            elif self.state == State.FORECAST:
                try:
                    if self.forecaster is None:
                        print("⚠️  Модель не загружена, пропускаем прогноз.")
                        self.state = State.DONE
                        continue

                    print("🔮 Создание прогноза...")
                    df = self.db.get_all_data(self.symbol)

                    if len(df) == 0:
                        raise ValueError("Нет данных в базе!")

                    # Делаем прогноз
                    self.forecast = self.forecaster.predict(df)

                    # Генерируем сигнал
                    self.signal = self.signal_generator.generate(self.forecast)

                    # Выводим результат
                    print("\n" + "="*60)
                    print(self.signal_generator.format_signal(self.signal, self.forecast))
                    print("="*60 + "\n")

                    # Переходим к уведомлениям
                    self.state = State.NOTIFY

                except Exception as e:
                    print(f"❌ Ошибка прогноза: {e}")
                    import traceback
                    traceback.print_exc()
                    self.state = State.ERROR

            elif self.state == State.NOTIFY:
                try:
                    if self.enable_notifications and self.notifier and self.forecast and self.signal:
                        print("📢 Отправка уведомления...")

                        # Формируем сообщение
                        message = self.signal_generator.format_signal(self.signal, self.forecast)

                        # Определяем приоритет по силе сигнала
                        if self.signal['signal'].startswith('STRONG'):
                            priority = 5
                            tags = ['chart_with_upwards_trend', 'fire']
                        elif self.signal['confidence'] == 'HIGH':
                            priority = 4
                            tags = ['chart_with_upwards_trend', 'zap']
                        else:
                            priority = 3
                            tags = ['chart', 'bell']

                        # Отправляем
                        self.notifier.send(
                            message=message,
                            title=f"🔮 {self.symbol} Прогноз",
                            priority=priority,
                            tags=tags
                        )
                    else:
                        print("ℹ️  Уведомления отключены")

                    self.state = State.DONE

                except Exception as e:
                    print(f"⚠️  Ошибка отправки уведомления: {e}")
                    # Не критичная ошибка, продолжаем
                    self.state = State.DONE

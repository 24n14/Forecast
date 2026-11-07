# fsm.py
from enum import Enum, auto
from ensemble import EnsembleForecaster
from signal_generator import SignalGenerator
from notifier import Notifier
from visualization import plot_forecast_bars
import os


class State(Enum):
    START = auto()
    DOWNLOAD = auto()
    FORECAST = auto()
    VISUALIZE = auto()
    NOTIFY = auto()
    DONE = auto()
    ERROR = auto()


class TradingFSM:
    def __init__(self, client, db, symbol, timeframe,
                 ntfy_topic=None, enable_notifications=True,
                 enable_visualization=True):
        self.client = client
        self.db = db
        self.symbol = symbol
        self.timeframe = timeframe

        # Визуализация
        self.enable_visualization = enable_visualization
        self.forecast_image_path = None

        # Загружаем модель прогнозирования
        try:
            self.forecaster = EnsembleForecaster()
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print("   Бот продолжит работу без прогнозов.")
            self.forecaster = None

        # Генератор сигналов
        self.signal_generator = SignalGenerator()

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
                    raw_df = self.client.fetch_candles(
                        symbol=self.symbol,
                        interval=self.timeframe,
                        limit=1000
                    )

                    if raw_df is not None and not raw_df.empty:
                        self.db.save_candles(raw_df, self.symbol, self.timeframe)
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
                    df = self.db.get_all_data(self.symbol, self.timeframe)

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

                    # Переходим к визуализации
                    if self.enable_visualization:
                        self.state = State.VISUALIZE
                    else:
                        self.state = State.NOTIFY

                except Exception as e:
                    print(f"❌ Ошибка прогноза: {e}")
                    import traceback
                    traceback.print_exc()
                    self.state = State.ERROR

            elif self.state == State.VISUALIZE:
                try:
                    print("🎨 Создание визуализации...")
                    df = self.db.get_all_data(self.symbol, self.timeframe)

                    # Создаём папку для графиков, если её нет
                    os.makedirs('forecasts', exist_ok=True)

                    # Путь к файлу
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    self.forecast_image_path = f'forecasts/{timestamp}_forecast_bars.png'

                    # Генерируем график
                    plot_forecast_bars(
                        history_df=df,
                        forecast=self.forecast,
                        save_path=self.forecast_image_path
                    )

                    self.state = State.NOTIFY

                except Exception as e:
                    print(f"⚠️  Ошибка визуализации: {e}")
                    import traceback
                    traceback.print_exc()
                    # Не критично, продолжаем без графика
                    self.forecast_image_path = None
                    self.state = State.NOTIFY

            elif self.state == State.NOTIFY:
                try:
                    if self.enable_notifications and self.notifier and self.forecast and self.signal:
                        print("📢 Отправка уведомления...")

                        # Формируем сообщение
                        message = self.signal_generator.format_signal(self.signal, self.forecast)

                        # Приоритет и теги уведомления зависят от режима волатильности
                        priority = self.signal['priority']
                        tags = self.signal['tags']

                        # Отправляем с графиком (если он есть)
                        if self.forecast_image_path and os.path.exists(self.forecast_image_path):
                            self.notifier.send_with_image(
                                message=message,
                                title=f"🔮 {self.symbol} Прогноз",
                                image_path=self.forecast_image_path,
                                priority=priority,
                                tags=tags
                            )
                        else:
                            # Отправляем без графика
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

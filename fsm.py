# fsm.py
from enum import Enum, auto
from ta_engine import add_features  # Оставляем этот импорт, он нужен для индикаторов


class State(Enum):
    START = auto()
    DOWNLOAD = auto()
    CALCULATE = auto()
    DONE = auto()
    ERROR = auto()


class TradingFSM:
    # ❌ БЫЛО: def __init__(self, client, db, symbol, config):
    # ✅ СТАЛО: передаём конкретные значения, а не весь конфиг
    def __init__(self, client, db, symbol, timeframe, history_days):
        self.client = client
        self.db = db
        self.symbol = symbol

        # Сохраняем параметры как простые переменные
        self.timeframe = timeframe  # Теперь это гарантированно строка "D"
        self.history_days = history_days

        self.state = State.START
        self.df = None

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
                        self.state = State.CALCULATE
                    else:
                        raise ValueError("Получен пустой DataFrame. Проверьте символ.")

                except Exception as e:
                    print(f"❌ Ошибка загрузки: {e}")
                    self.state = State.ERROR

            elif self.state == State.CALCULATE:
                try:
                    print("📊 Расчет индикаторов...")
                    # ✅ Используем self.history_days вместо self.config.HISTORY_DAYS
                    self.df = self.db.get_data(self.symbol, days=self.history_days)

                    if len(self.df) == 0:
                        raise ValueError("Нет данных в базе! Сначала проверьте загрузку.")

                    self.df = self.df.sort_values('time').reset_index(drop=True)
                    self.df = add_features(self.df)

                    print(f"✅ Готово. Строк: {len(self.df)}, колонок: {len(self.df.columns)}")
                    print(f"📋 Новые колонки: {list(self.df.columns)[-5:]}")
                    self.state = State.DONE

                except Exception as e:
                    print(f"❌ Ошибка расчета: {e}")
                    import traceback
                    traceback.print_exc()
                    self.state = State.ERROR

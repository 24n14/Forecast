from enum import Enum, auto
# ГЛАВНОЕ ИСПРАВЛЕНИЕ: импортируем функцию add_features из ta_engine
from ta_engine import add_features


class State(Enum):
    START = auto()
    DOWNLOAD = auto()
    CALCULATE = auto()
    DONE = auto()
    ERROR = auto()


class TradingFSM:
    def __init__(self, client, db, symbol, config):
        self.client = client
        self.db = db
        self.symbol = symbol
        self.config = config
        self.state = State.START
        self.df = None

    def run(self):
        while self.state != State.DONE and self.state != State.ERROR:
            if self.state == State.START:
                self.state = State.DOWNLOAD

            elif self.state == State.DOWNLOAD:
                try:
                    print("📥 Загрузка свечей...")
                    # Важно: limit=1000 может не покрыть 180 дней, если были дыры в данных.
                    # Для продакшена тут нужна логика дозагрузки, но для старта хватит.
                    raw_df = self.client.fetch_candles(self.symbol, self.config.TIMEFRAME, limit=1000)
                    self.db.save_candles(raw_df)
                    self.state = State.CALCULATE
                except Exception as e:
                    print(f"❌ Ошибка загрузки: {e}")
                    self.state = State.ERROR

            elif self.state == State.CALCULATE:
                try:
                    print("📊 Расчет индикаторов...")
                    # Берем данные из БД
                    self.df = self.db.get_data(self.symbol, days=self.config.HISTORY_DAYS)

                    if len(self.df) == 0:
                        raise ValueError("Нет данных в базе! Сначала проверьте загрузку.")

                    # Сортировка обязательна: TA-LIB считает по порядку строк
                    self.df = self.df.sort_values('time').reset_index(drop=True)

                    # ТЕПЕРЬ ЭТО РАБОТАЕТ: функция импортирована сверху
                    self.df = add_features(self.df)

                    print(f"✅ Готово. Строк: {len(self.df)}, колонок: {len(self.df.columns)}")
                    print(f"📋 Новые колонки: {list(self.df.columns)[-5:]}")  # Покажем последние 5 колонок
                    self.state = State.DONE
                except Exception as e:
                    print(f"❌ Ошибка расчета: {e}")
                    import traceback
                    traceback.print_exc()  # Чтобы видеть точную строку ошибки
                    self.state = State.ERROR

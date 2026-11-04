#!/usr/bin/env python3
# main.py
# Главная точка входа бота с поддержкой однократного и циклического запуска
import argparse
from config import (
    SYMBOL, TIMEFRAME, DB_DSN, HISTORY_DAYS,
    NTFY_TOPIC, ENABLE_NOTIFICATIONS,
    SIGNAL_THRESHOLD_STRONG, SIGNAL_THRESHOLD_WEAK,
    SCHEDULE_ENABLED, SCHEDULE_TIME, SCHEDULE_TIMEZONE
)
from bybit_client import BybitClient
from db_layer import DatabaseLayer
from fsm import TradingFSM
from scheduler import BotScheduler


def run_bot():
    """Запуск одного цикла работы бота."""
    print(f"🚀 Запуск бота: {SYMBOL}, TF={TIMEFRAME}")

    client = BybitClient()
    db = DatabaseLayer(DB_DSN)

    fsm = TradingFSM(
        client=client,
        db=db,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        history_days=HISTORY_DAYS,
        ntfy_topic=NTFY_TOPIC,
        enable_notifications=ENABLE_NOTIFICATIONS,
        signal_threshold_strong=SIGNAL_THRESHOLD_STRONG,
        signal_threshold_weak=SIGNAL_THRESHOLD_WEAK
    )

    fsm.run()


def main():
    parser = argparse.ArgumentParser(description="Forecast Bot - прогнозирование криптовалют")
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Запустить по расписанию (по умолчанию: однократный запуск)"
    )
    parser.add_argument(
        "--schedule-time",
        type=str,
        help=f"Время запуска в формате HH:MM (по умолчанию: {SCHEDULE_TIME})"
    )

    args = parser.parse_args()

    # Определяем режим работы
    if args.scheduled or SCHEDULE_ENABLED:
        # Циклический режим
        schedule_time = args.schedule_time if args.schedule_time else SCHEDULE_TIME
        scheduler = BotScheduler(
            run_func=run_bot,
            schedule_time=schedule_time,
            timezone=SCHEDULE_TIMEZONE
        )
        scheduler.run_scheduled()
    else:
        # Однократный запуск
        run_bot()


if __name__ == "__main__":
    main()

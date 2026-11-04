# scheduler.py
# Модуль для циклического запуска бота по расписанию
import time
import schedule
from datetime import datetime


class BotScheduler:
    """Управляет циклическим запуском бота."""

    def __init__(self, run_func, schedule_time="09:00", timezone="UTC"):
        """
        Args:
            run_func: функция для запуска бота (без аргументов)
            schedule_time: время запуска в формате "HH:MM"
            timezone: временная зона (пока только UTC, можно расширить)
        """
        self.run_func = run_func
        self.schedule_time = schedule_time
        self.timezone = timezone
        self.is_running = False

    def job_wrapper(self):
        """Обёртка для безопасного выполнения задачи."""
        print(f"\n{'='*60}")
        print(f"⏰ Запуск задачи: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        try:
            self.run_func()
            print(f"\n✅ Задача завершена успешно")
        except Exception as e:
            print(f"\n❌ Ошибка выполнения задачи: {e}")
            import traceback
            traceback.print_exc()

    def run_once(self):
        """Запустить бота один раз."""
        print("🚀 Однократный запуск...")
        self.job_wrapper()

    def run_scheduled(self):
        """Запустить планировщик с заданным расписанием."""
        print(f"📅 Планировщик запущен")
        print(f"⏰ Расписание: каждый день в {self.schedule_time} ({self.timezone})")
        print(f"⏳ Следующий запуск: {schedule.next_run()}")
        print(f"   Ctrl+C для остановки\n")

        # Настраиваем расписание
        schedule.every().day.at(self.schedule_time).do(self.job_wrapper)

        self.is_running = True

        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(30)  # Проверка каждые 30 секунд

        except KeyboardInterrupt:
            print("\n\n⚠️  Получен сигнал остановки (Ctrl+C)")
            self.stop()

    def stop(self):
        """Остановить планировщик."""
        self.is_running = False
        schedule.clear()
        print("🛑 Планировщик остановлен")


if __name__ == "__main__":
    # Тест
    def test_job():
        print("Выполняется тестовая задача...")
        time.sleep(2)
        print("Задача завершена")

    scheduler = BotScheduler(run_func=test_job, schedule_time="00:00")
    print("Тест однократного запуска:")
    scheduler.run_once()

    print("\nДля теста планировщика запустите:")
    print("  python scheduler.py --scheduled")

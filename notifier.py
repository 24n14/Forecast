# notifier.py
# Модуль для отправки уведомлений на ntfy.sh
import requests


class Notifier:
    """Отправляет уведомления на ntfy.sh."""

    def __init__(self, topic):
        """
        Args:
            topic: название топика ntfy.sh (например, 'forecaster_24n14')
        """
        self.topic = topic
        self.base_url = "https://ntfy.sh"

    def send(self, message, title=None, priority=None, tags=None):
        """
        Отправляет уведомление.

        Args:
            message: текст сообщения
            title: заголовок (опционально)
            priority: приоритет (1-5, default 3)
            tags: список эмодзи-тегов, например ['chart_with_upwards_trend', 'bell']

        Returns:
            bool: True если отправлено успешно
        """
        try:
            # Используем JSON API для корректной работы с unicode
            payload = {"topic": self.topic, "message": message}

            if title:
                payload["title"] = title
            if priority:
                payload["priority"] = priority
            if tags:
                payload["tags"] = tags

            response = requests.post(
                self.base_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                print(f"✅ Уведомление отправлено на {self.topic}")
                return True
            else:
                print(f"⚠️  Ошибка отправки: {response.status_code} {response.text}")
                return False

        except Exception as e:
            print(f"❌ Ошибка отправки уведомления: {e}")
            return False


if __name__ == "__main__":
    # Тест
    notifier = Notifier("forecaster_24n14")
    notifier.send(
        message="Тестовое сообщение от бота прогнозирования",
        title="🤖 Forecast Bot",
        priority=3,
        tags=['robot', 'test']
    )

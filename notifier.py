# notifier.py
# Модуль для отправки уведомлений на ntfy.sh
import requests
import os


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

    def send_with_image(self, message, title=None, image_path=None, priority=None, tags=None):
        """
        Отправляет прогноз двумя сообщениями в топик:
          1) картинку (PUT с бинарными данными — так работает ntfy.sh);
          2) текст прогноза (JSON).
        Текст нельзя прикрепить к бинарной загрузке: тело PUT занято файлом,
        а заголовки не поддерживают переводы строк и emoji.

        Args:
            message: текст сообщения
            title: заголовок (опционально)
            image_path: путь к PNG/JPG файлу
            priority: приоритет (1-5, default 3)
            tags: список эмодзи-тегов

        Returns:
            bool: True если текст отправлен успешно
        """
        if not image_path or not os.path.exists(image_path):
            print(f"⚠️  Изображение не найдено: {image_path}")
            return self.send(message, title, priority, tags)

        # 1. Картинка отдельным сообщением
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()

            headers = {
                # Имя файла -> ntfy покажет картинку прямо в уведомлении
                'Filename': os.path.basename(image_path),
                # Только ASCII: заголовки HTTP не принимают emoji
                'Title': f"{self.topic} chart",
            }
            if priority:
                headers['Priority'] = str(priority)

            response = requests.put(
                f"{self.base_url}/{self.topic}",
                data=image_data,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                print(f"✅ График отправлен на {self.topic}")
            else:
                print(f"⚠️  Ошибка отправки графика: {response.status_code} {response.text}")

        except Exception as e:
            print(f"⚠️  Ошибка отправки графика: {e}")

        # 2. Текст прогноза — через JSON, unicode поддерживается полностью
        return self.send(message, title, priority, tags)


if __name__ == "__main__":
    # Тест
    notifier = Notifier("forecaster_24n14")
    notifier.send(
        message="Тестовое сообщение от бота прогнозирования",
        title="🤖 Forecast Bot",
        priority=3,
        tags=['robot', 'test']
    )

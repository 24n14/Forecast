# signal_generator.py
# Модуль для генерации торговых сигналов на основе прогноза


class SignalGenerator:
    """Генерирует сигналы на основе квантильного прогноза."""

    def __init__(self, threshold_strong=10.0, threshold_weak=5.0):
        """
        Args:
            threshold_strong: порог для сильного сигнала (в %)
            threshold_weak: порог для слабого сигнала (в %)
        """
        self.threshold_strong = threshold_strong
        self.threshold_weak = threshold_weak

    def generate(self, forecast):
        """
        Генерирует сигнал на основе прогноза.

        Args:
            forecast: dict с прогнозом от Forecaster.predict()

        Returns:
            dict: {
                'signal': 'STRONG_BULLISH' | 'WEAK_BULLISH' | 'NEUTRAL' | 'WEAK_BEARISH' | 'STRONG_BEARISH',
                'confidence': 'HIGH' | 'MEDIUM' | 'LOW',
                'summary': str описание сигнала,
                'median_change': float изменение медианы в %,
                'risk_range': (float, float) диапазон [q05, q95],
            }
        """
        q05 = forecast['quantiles'][0.05]['change_pct']
        q50 = forecast['quantiles'][0.5]['change_pct']
        q95 = forecast['quantiles'][0.95]['change_pct']

        # Определяем направление по медиане
        if q50 >= self.threshold_strong:
            signal = 'STRONG_BULLISH'
            emoji = '🟢📈'
        elif q50 >= self.threshold_weak:
            signal = 'WEAK_BULLISH'
            emoji = '🟢'
        elif q50 <= -self.threshold_strong:
            signal = 'STRONG_BEARISH'
            emoji = '🔴📉'
        elif q50 <= -self.threshold_weak:
            signal = 'WEAK_BEARISH'
            emoji = '🔴'
        else:
            signal = 'NEUTRAL'
            emoji = '⚪'

        # Оцениваем уверенность: если q05 и q95 смотрят в одну сторону — высокая
        if q05 > 0 and q95 > 0:
            confidence = 'HIGH'
            confidence_emoji = '🔥'
        elif q05 < 0 and q95 < 0:
            confidence = 'HIGH'
            confidence_emoji = '🔥'
        elif abs(q50) >= self.threshold_strong:
            confidence = 'MEDIUM'
            confidence_emoji = '⚡'
        else:
            confidence = 'LOW'
            confidence_emoji = '💡'

        # Собираем описание
        summary_parts = [
            f"{emoji} Сигнал: {signal}",
            f"{confidence_emoji} Уверенность: {confidence}",
            f"Медиана: {q50:+.2f}%",
            f"Диапазон: [{q05:+.2f}%, {q95:+.2f}%]"
        ]

        return {
            'signal': signal,
            'confidence': confidence,
            'summary': '\n'.join(summary_parts),
            'median_change': q50,
            'risk_range': (q05, q95),
            'emoji': emoji,
            'confidence_emoji': confidence_emoji,
        }

    def format_signal(self, signal_data, forecast):
        """Форматирует сигнал и прогноз в читаемое сообщение."""
        lines = []

        # Заголовок
        lines.append(f"{signal_data['emoji']} {signal_data['signal']}")
        lines.append(f"{signal_data['confidence_emoji']} Уверенность: {signal_data['confidence']}")
        lines.append("")

        # Текущая цена
        current = forecast['current_price']
        date = forecast['current_date'].strftime('%Y-%m-%d')
        lines.append(f"💰 Текущая цена: ${current:,.2f}")
        lines.append(f"📅 Дата: {date}")
        lines.append(f"⏳ Горизонт: {forecast['forecast_horizon']} дней")
        lines.append("")

        # Прогноз по квантилям
        lines.append("📊 Прогноз:")
        for q, data in forecast['quantiles'].items():
            q_label = f"q{int(q*100):02d}"
            change = data['change_pct']
            target = data['target_price']
            emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            lines.append(f"  {emoji} {q_label}: {change:+6.2f}% → ${target:,.2f}")

        return '\n'.join(lines)


if __name__ == "__main__":
    # Тест
    import pandas as pd

    test_forecast = {
        'current_price': 78409.90,
        'current_date': pd.Timestamp('2026-11-04'),
        'forecast_horizon': 30,
        'quantiles': {
            0.05: {'change_pct': -14.57, 'target_price': 66987.29},
            0.5: {'change_pct': 3.46, 'target_price': 81121.86},
            0.95: {'change_pct': 15.36, 'target_price': 90455.98},
        }
    }

    sg = SignalGenerator()
    signal = sg.generate(test_forecast)
    print(sg.format_signal(signal, test_forecast))

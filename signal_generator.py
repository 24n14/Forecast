# signal_generator.py
# Сигнал по режиму волатильности.
#
# Модель (HAR) прогнозирует не направление, а размах движения цены за горизонт.
# Поэтому сигнал говорит о режиме рынка и о вероятности сильного движения,
# а направление явно помечается как непрогнозируемое.

REGIME_INFO = {
    'CALM': {
        'emoji': '🟦',
        'title': 'СПОКОЙНЫЙ РЕЖИМ',
        'hint': 'Ожидается движение меньше обычного: коридор узкий.',
        'priority': 2,
        'tags': ['zzz'],
    },
    'NORMAL': {
        'emoji': '⚪',
        'title': 'ОБЫЧНЫЙ РЕЖИМ',
        'hint': 'Волатильность в пределах нормы.',
        'priority': 3,
        'tags': ['chart'],
    },
    'VOLATILE': {
        'emoji': '🟧⚡',
        'title': 'ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ',
        'hint': 'Повышен риск сильного движения в любую сторону.',
        'priority': 4,
        'tags': ['warning', 'zap'],
    },
}


class SignalGenerator:
    """Формирует сигнал режима волатильности из прогноза HarForecaster."""

    def generate(self, forecast):
        """
        Returns:
            dict: {'signal': 'CALM' | 'NORMAL' | 'VOLATILE', 'emoji', 'title', 'hint', 'priority', 'tags'}
        """
        regime = forecast.get('regime', 'NORMAL')
        return {'signal': regime, **REGIME_INFO[regime]}

    def format_signal(self, signal_data, forecast):
        """Форматирует сигнал и прогноз в читаемое сообщение."""
        current = forecast['current_price']
        date = forecast['current_date'].strftime('%Y-%m-%d %H:%M')
        q = forecast['quantiles']
        lo, hi = q[0.05], q[0.95]

        lines = [
            f"{signal_data['emoji']} {signal_data['title']}",
            signal_data['hint'],
            "",
            f"💰 Текущая цена: ${current:,.2f}",
            f"📅 Данные на: {date}",
            f"⏳ Горизонт: {forecast['forecast_horizon']} дней",
        ]

        if 'sigma_pct' in forecast:
            rank = forecast.get('sigma_percentile')
            if rank is None:
                rank_txt = ""
            elif rank < 50:
                rank_txt = f" (ниже, чем в {100 - rank:.0f}% истории)"
            else:
                rank_txt = f" (выше, чем в {rank:.0f}% истории)"
            lines.append(f"🌊 Ожидаемая волатильность σ: {forecast['sigma_pct']:.1f}%{rank_txt}")

        lines += [
            "",
            "📏 Коридор 90%:",
            f"  🔻 {lo['change_pct']:+.1f}% → ${lo['target_price']:,.0f}",
            f"  🔺 {hi['change_pct']:+.1f}% → ${hi['target_price']:,.0f}",
        ]

        if forecast.get('move_probs'):
            lines += ["", "🎯 Вероятность движения в любую сторону:"]
            for threshold, p in forecast['move_probs'].items():
                lines.append(f"  больше {threshold:g}%: {p:.0%}")

        lines += ["", "🧭 Направление: не прогнозируется (модель оценивает только размах)."]
        return '\n'.join(lines)


if __name__ == "__main__":
    import pandas as pd

    test_forecast = {
        'current_price': 75864.0,
        'current_date': pd.Timestamp('2026-11-07'),
        'forecast_horizon': 7,
        'sigma_pct': 6.3,
        'sigma_percentile': 48,
        'regime': 'NORMAL',
        'move_probs': {5.0: 0.41, 10.0: 0.14},
        'quantiles': {
            0.05: {'change_pct': -9.16, 'target_price': 68914.86},
            0.5: {'change_pct': 0.33, 'target_price': 76117.85},
            0.95: {'change_pct': 12.58, 'target_price': 85407.66},
        },
    }
    sg = SignalGenerator()
    print(sg.format_signal(sg.generate(test_forecast), test_forecast))

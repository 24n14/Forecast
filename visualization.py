# visualization.py
# Визуализация прогноза: гистограмма вероятности роста/падения по дням
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Без GUI: бот работает в фоне / по расписанию
import matplotlib.pyplot as plt

Z90 = 1.645  # z-оценка 95-го квантиля нормального распределения

def _quantile_params(forecast):
    """Медиана и раздельные σ вниз/вверх, чтобы сохранить асимметрию прогноза."""
    q05 = forecast['quantiles'][0.05]['change_pct']
    q50 = forecast['quantiles'][0.5]['change_pct']
    q95 = forecast['quantiles'][0.95]['change_pct']
    sigma_down = max(q50 - q05, 1e-6) / Z90
    sigma_up = max(q95 - q50, 1e-6) / Z90
    return q50, sigma_down, sigma_up

# ---------------------------------------------------------------------------
# Гистограмма бычьей/медвежьей вероятности по дням
# ---------------------------------------------------------------------------
BULL = '#5fe39a'       # рост (пара подобрана: ΔE при дейтеранопии 25, контраст с фоном ≥ 3:1)
BEAR = '#c8373d'       # падение
BG_PAGE, BG_PLOT = '#0A0D12', '#0F131C'
INK, INK_MUTED = '#F0EFEC', '#9A9FAB'
MONTHS_RU = ['янв', 'фев', 'мар', 'апр', 'мая', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']


def _dim(color, k=0.55):
    """Сплошной приглушённый цвет (смесь с фоном) — без прозрачности, чтобы линии не просвечивали."""
    c, bg = np.array(matplotlib.colors.to_rgb(color)), np.array(matplotlib.colors.to_rgb(BG_PLOT))
    return tuple(k * c + (1 - k) * bg)


def _ru_date(d):
    return f"{d.day:02d} {MONTHS_RU[d.month - 1]}"


def _normal_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x) / np.sqrt(2)))


def daily_up_probability(forecast):
    """P(цена выше текущей) на каждый день горизонта.

    Та же модель траекторий, что и у облака: медиана движется линейно,
    разброс растёт как √t, вниз и вверх — свои σ (split-normal по q05/q50/q95).
    """
    horizon = forecast['forecast_horizon']
    q50, sigma_down, sigma_up = _quantile_params(forecast)
    t = np.arange(1, horizon + 1)
    drift = q50 * t / horizon
    scale = np.sqrt(t / horizon)
    # change = drift + z·σ, σ = σ_up при z > 0 и σ_down при z < 0 → P(change > 0)
    sigma = np.where(drift >= 0, sigma_down, sigma_up) * scale
    return _normal_cdf(drift / sigma)


def past_daily_changes(history_df, current_date, days=7):
    """Изменение цены (в %) за последние `days` завершённых календарных дней."""
    h = history_df.sort_values('time')
    daily_close = h.set_index('time')['close'].resample('D').last().dropna()
    daily_close = daily_close[daily_close.index < pd.Timestamp(current_date).normalize()]
    change = daily_close.pct_change().dropna() * 100
    return change.tail(days)


def plot_forecast_bars(history_df, forecast, save_path='forecast_bars.png',
                       past_days=7, symbol='BTC/USDT'):
    """14 столбцов по дням: 7 прошлых (факт) + 7 прогноза (вероятность роста/падения).

    В каждом столбце зелёная часть сверху — доля роста, красная снизу — доля падения.
    """
    past = past_daily_changes(history_df, forecast['current_date'], past_days)
    p_up = daily_up_probability(forecast)
    today = pd.Timestamp(forecast['current_date']).normalize()

    past_dates = list(past.index)
    fut_dates = [today + pd.Timedelta(days=d) for d in range(1, len(p_up) + 1)]
    up = np.r_[(past.values > 0).astype(float), p_up]
    n_past = len(past)
    x = np.arange(len(up))

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_PAGE)
    ax.set_facecolor(BG_PLOT)
    width = 0.78

    for i in x:
        down = 1 - up[i]
        # Красная часть снизу, зелёная сверху; зазор 2px цветом фона между частями
        # Прошлые дни приглушены: это факт, а не прогноз
        bull, bear = (_dim(BULL), _dim(BEAR)) if i < n_past else (BULL, BEAR)
        if down > 0:
            ax.bar(i, down * 100, width, bottom=0, color=bear,
                   edgecolor=BG_PLOT, linewidth=2, zorder=3)
        if up[i] > 0:
            ax.bar(i, up[i] * 100, width, bottom=down * 100, color=bull,
                   edgecolor=BG_PLOT, linewidth=2, zorder=3)

        if i < n_past:
            ch = past.values[i]
            arrow = '▲' if ch > 0 else '▼'
            ax.text(i, 50, f"{arrow}\n{ch:+.1f}%", ha='center', va='center', fontsize=10,
                    color='#0b0b0b' if ch > 0 else 'white', fontweight='bold', zorder=5)
        else:
            # Подписи долей внутри частей, если часть достаточно высокая
            if up[i] >= 0.12:
                ax.text(i, (1 - up[i] / 2) * 100, f"{up[i]:.0%}", ha='center', va='center',
                        fontsize=11, color='#0b0b0b', fontweight='bold', zorder=5)
            if down >= 0.12:
                ax.text(i, down / 2 * 100, f"{down:.0%}", ha='center', va='center',
                        fontsize=11, color='white', fontweight='bold', zorder=5)

    # Граница "сегодня" и линия 50%
    ax.axvline(n_past - 0.5, color='#38BDF8', linestyle=':', linewidth=1.5, alpha=0.8, zorder=4)
    ax.axhline(50, color=INK_MUTED, linestyle='--', linewidth=1, alpha=0.5, zorder=2)
    ax.text(n_past - 0.5, -0.07, f'сегодня\n{_ru_date(today)}', transform=ax.get_xaxis_transform(),
            ha='center', va='top', color='#38BDF8', fontsize=9, fontweight='bold')
    ax.text(n_past / 2 - 0.5, 103, 'ФАКТ: рост / падение за день', ha='center', va='bottom',
            color=INK_MUTED, fontsize=11)
    ax.text(n_past + len(p_up) / 2 - 0.5, 103, 'ПРОГНОЗ: вероятность, что цена выше / ниже текущей',
            ha='center', va='bottom', color=INK, fontsize=11)

    labels = [_ru_date(d) for d in past_dates + fut_dates]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK, fontsize=9)
    for lbl in ax.get_xticklabels()[:n_past]:
        lbl.set_color(INK_MUTED)
    ax.set_xlim(-0.6, len(up) - 0.4)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(True, axis='y', alpha=0.08, color='white', linewidth=0.5, zorder=0)

    horizon = forecast['forecast_horizon']
    ax.set_title(f'{symbol} — рост или падение по дням: 7 дней факта + прогноз на {horizon} дней',
                 color='white', fontsize=14, fontweight='bold', pad=28)

    legend = ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BULL, label='Рост (бычий)'),
                                plt.Rectangle((0, 0), 1, 1, color=BEAR, label='Падение (медвежий)')],
                       loc='upper left', bbox_to_anchor=(0, -0.08), ncol=2, fontsize=10,
                       facecolor=BG_PAGE, edgecolor='none', labelcolor=INK, frameon=False)

    q = forecast['quantiles']
    price = forecast['current_price']
    info = (rf"Текущая: \${price:,.0f}   |   Коридор 90% на {horizon} дн.: "
            rf"\${q[0.05]['target_price']:,.0f} — \${q[0.95]['target_price']:,.0f}   |   "
            f"P(рост к дню {horizon}): {p_up[-1]:.0%}")
    fig.text(0.98, 0.025, info, ha='right', va='bottom', color=INK, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#161D2B', edgecolor='#38BDF8', linewidth=1.2))

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(save_path, dpi=150, facecolor=BG_PAGE, edgecolor='none')
    plt.close(fig)
    print(f"📊 График сохранён: {save_path}")
    return save_path

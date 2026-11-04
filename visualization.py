# visualization.py
# Визуализация прогноза в виде точечного облака сценариев
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Без GUI: бот работает в фоне / по расписанию
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

Z90 = 1.645  # z-оценка 95-го квантиля нормального распределения


def _quantile_params(forecast):
    """Медиана и раздельные σ вниз/вверх, чтобы сохранить асимметрию прогноза."""
    q05 = forecast['quantiles'][0.05]['change_pct']
    q50 = forecast['quantiles'][0.5]['change_pct']
    q95 = forecast['quantiles'][0.95]['change_pct']
    sigma_down = max(q50 - q05, 1e-6) / Z90
    sigma_up = max(q95 - q50, 1e-6) / Z90
    return q50, sigma_down, sigma_up


def generate_scenario_samples(forecast, n_samples=300, rng=None):
    """
    Генерирует траектории цены на forecast_horizon дней.

    Каждая траектория — случайное блуждание (броуновское движение), поэтому
    разброс растёт как √t: завтра почти рядом с текущей ценой, к концу
    горизонта — во всю ширину прогноза. Отклонения вниз и вверх
    масштабируются разными σ (split-normal), так что на последнем дне
    5-й и 95-й перцентили облака совпадают с q05 и q95 модели.

    Returns:
        np.array shape (n_samples, horizon + 1): цены, столбец 0 — текущая цена
    """
    rng = rng or np.random.default_rng()
    current_price = forecast['current_price']
    horizon = forecast['forecast_horizon']
    q50, sigma_down, sigma_up = _quantile_params(forecast)

    t = np.arange(horizon + 1)
    # Стандартное броуновское движение, нормированное так, что z[:, H] ~ N(0, 1)
    steps = rng.standard_normal((n_samples, horizon)) / np.sqrt(horizon)
    z = np.concatenate([np.zeros((n_samples, 1)), np.cumsum(steps, axis=1)], axis=1)

    drift = q50 * t / horizon                       # медиана движется линейно
    change = drift + np.where(z > 0, z * sigma_up, z * sigma_down)
    return current_price * (1 + change / 100)


def quantile_band(forecast):
    """Линии q05 / q50 / q95 по дням — та же форма √t, что и у облака."""
    current_price = forecast['current_price']
    horizon = forecast['forecast_horizon']
    q50, sigma_down, sigma_up = _quantile_params(forecast)

    t = np.arange(horizon + 1)
    drift = q50 * t / horizon
    spread = Z90 * np.sqrt(t / horizon)
    to_price = lambda pct: current_price * (1 + pct / 100)
    return (to_price(drift - spread * sigma_down),
            to_price(drift),
            to_price(drift + spread * sigma_up))


def plot_forecast_cloud(history_df, forecast, save_path='forecast_cloud.png',
                        n_samples=300, history_days=60, symbol='BTC/USDT'):
    """Рисует историю цены и облако сценариев прогноза, сохраняет PNG."""
    history_slice = history_df.sort_values('time').tail(history_days)

    current_date = forecast['current_date']
    current_price = forecast['current_price']
    horizon = forecast['forecast_horizon']
    forecast_dates = pd.date_range(current_date, periods=horizon + 1, freq='D')

    trajectories = generate_scenario_samples(forecast, n_samples)
    q05_line, q50_line, q95_line = quantile_band(forecast)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0A0D12')
    ax.set_facecolor('#0F131C')

    # Облако: все точки одним вызовом; яркость падает по мере удаления от медианы
    xs = np.tile(mdates.date2num(forecast_dates[1:]), n_samples)
    ys = trajectories[:, 1:].ravel()
    half_width = np.tile(np.maximum(q95_line[1:] - q05_line[1:], 1e-9) / 2, n_samples)
    dist = np.abs(ys - np.tile(q50_line[1:], n_samples)) / half_width
    colors = np.zeros((len(ys), 4))
    colors[:, :3] = matplotlib.colors.to_rgb('#6EE7B7')
    colors[:, 3] = np.clip(0.35 * (1 - 0.6 * np.minimum(dist, 1.5)), 0.04, 0.35)
    ax.scatter(xs, ys, c=colors, s=7, linewidths=0, zorder=3)

    # Зона 90% и квантильные линии
    ax.fill_between(forecast_dates, q05_line, q95_line,
                    color='#6EE7B7', alpha=0.08, zorder=1, label='Зона 90%')
    ax.plot(forecast_dates, q50_line, color='#6EE7B7', linewidth=3,
            zorder=4, label='Медиана (q50)')
    ax.plot(forecast_dates, q05_line, color='#E9A568', linewidth=2,
            linestyle='--', zorder=4, label='q05 / q95')
    ax.plot(forecast_dates, q95_line, color='#E9A568', linewidth=2,
            linestyle='--', zorder=4)

    # История и точка "сегодня"
    ax.plot(history_slice['time'], history_slice['close'],
            color='#3B6DFF', linewidth=2.5, zorder=5, label='История')
    ax.scatter([current_date], [current_price], color='#38BDF8', s=100,
               zorder=10, edgecolors='white', linewidths=1.5)
    ax.axvline(current_date, color='#38BDF8', linestyle=':', linewidth=1.5, alpha=0.5)

    # Оформление (символ $ экранируем: иначе matplotlib включает режим формул)
    ax.set_title(f'{symbol} — прогноз на {horizon} дней | {n_samples} сценариев',
                 color='white', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Дата', color='white', fontsize=11)
    ax.set_ylabel(r'Цена, \$', color='white', fontsize=11)
    ax.tick_params(colors='white', labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: rf'\${x:,.0f}'))
    ax.grid(True, alpha=0.15, color='white', linewidth=0.5)
    ax.text(current_date, 1.0, ' Сегодня', transform=ax.get_xaxis_transform(),
            color='#38BDF8', fontsize=10, va='top', ha='left')

    legend = ax.legend(loc='upper left', fontsize=10, facecolor='#161D2B',
                       edgecolor='#38BDF8', labelcolor='white', framealpha=0.9)
    legend.get_frame().set_linewidth(1.5)

    q50_pct = forecast['quantiles'][0.5]['change_pct']
    info_text = (
        rf"Текущая: \${current_price:,.0f}" "\n"
        rf"Медиана: \${q50_line[-1]:,.0f} ({q50_pct:+.1f}%)" "\n"
        rf"Диапазон 90%: \${q05_line[-1]:,.0f} — \${q95_line[-1]:,.0f}"
    )
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=9,
            color='white', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#161D2B',
                      edgecolor='#38BDF8', alpha=0.9, linewidth=1.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0A0D12', edgecolor='none')
    plt.close(fig)

    print(f"📊 График сохранён: {save_path}")
    return save_path


if __name__ == "__main__":
    from db_layer import DatabaseLayer
    from forecaster import Forecaster
    from config import DB_DSN, SYMBOL

    df = DatabaseLayer(DB_DSN).get_all_data(SYMBOL)
    forecast = Forecaster().predict(df)

    # Проверка калибровки: перцентили облака на последнем дне должны совпасть с q05/q50/q95
    sims = generate_scenario_samples(forecast, 20000)[:, -1]
    p05, p50, p95 = np.percentile(sims, [5, 50, 95])
    print(f"Облако, день {forecast['forecast_horizon']}: p05={p05:,.0f} p50={p50:,.0f} p95={p95:,.0f}")
    print(f"Модель:            q05={forecast['quantiles'][0.05]['target_price']:,.0f} "
          f"q50={forecast['quantiles'][0.5]['target_price']:,.0f} "
          f"q95={forecast['quantiles'][0.95]['target_price']:,.0f}")

    plot_forecast_cloud(df, forecast, save_path='forecasts/test_forecast_cloud.png')

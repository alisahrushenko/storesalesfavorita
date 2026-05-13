import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

import tensorflow as tf
from tensorflow.keras.models import load_model

# ══════════════════════════════════════════════
#  НАЛАШТУВАННЯ СТОРІНКИ
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Sales Forecasting | LSTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════
#  СТИЛІ
# ══════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
        color: #e8eaf0;
    }

    /* Заголовок */
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
        border: 1px solid #2d3348;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4f8ef7, #a855f7, #ec4899);
    }
    .main-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #f0f4ff;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8892a4;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 300;
    }

    /* Вкладки */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f2e;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #2d3348;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #8892a4;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #2d3348 !important;
        color: #4f8ef7 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    /* Метрики */
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #4f8ef7; }
    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #8892a4;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #4f8ef7;
    }
    .metric-value.good { color: #34d399; }
    .metric-value.warn { color: #fbbf24; }

    /* Інфо блоки */
    .info-box {
        background: #1a1f2e;
        border-left: 3px solid #4f8ef7;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #c8d0e0;
    }

    /* Секції */
    .section-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #4f8ef7;
        letter-spacing: 0.5px;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2d3348;
    }

    /* Таблиця */
    .dataframe {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    /* Слайдер */
    .stSlider > div { color: #8892a4; }

    /* Приховати зайві елементи */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 3rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  ЗАВАНТАЖЕННЯ ДАНИХ
# ══════════════════════════════════════════════
@st.cache_data
def load_data():
    weekly   = pd.read_csv('weekly_sales.csv', parse_dates=['Date'])
    metrics  = pd.read_csv('model_metrics.csv')
    forecast = pd.read_csv('forecast_26weeks.csv', parse_dates=['Date'])
    oil      = pd.read_csv('oil.csv', parse_dates=['date'])
    holidays = pd.read_csv('holidays_events.csv', parse_dates=['date'])
    stores   = pd.read_csv('stores.csv')
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    return weekly, metrics, forecast, oil, holidays, stores, config

@st.cache_resource
def load_model_and_scaler():
    model  = load_model('lstm_sales_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Графік стиль
def set_plot_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor':  '#1a1f2e',
        'axes.facecolor':    '#1a1f2e',
        'axes.edgecolor':    '#2d3348',
        'axes.labelcolor':   '#8892a4',
        'xtick.color':       '#8892a4',
        'ytick.color':       '#8892a4',
        'grid.color':        '#2d3348',
        'grid.linewidth':    0.5,
        'text.color':        '#e8eaf0',
        'font.family':       'monospace',
    })

set_plot_style()

# ══════════════════════════════════════════════
#  ЗАГОЛОВОК
# ══════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🧠 Sales Forecasting System</h1>
    <p>Система прогнозування продажів на основі нейромережі LSTM &nbsp;·&nbsp;
       Store Sales — Corporación Favorita &nbsp;·&nbsp; 2013–2017</p>
</div>
""", unsafe_allow_html=True)

# Завантаження
try:
    weekly, metrics_df, forecast_df, oil_df, holidays_df, stores_df, config = load_data()
    model, scaler = load_model_and_scaler()
    data_loaded = True
except Exception as e:
    st.error(f'❌ Помилка завантаження даних: {e}')
    st.info('Переконайся що всі файли є в репозиторії.')
    data_loaded = False
    st.stop()

# ══════════════════════════════════════════════
#  ВКЛАДКИ
# ══════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA — Аналіз даних",
    "🔮 Прогнозування",
    "📈 Порівняння моделей",
    "📋 Метрики"
])

# ══════════════════════════════════════════════
#  ВКЛАДКА 1: EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">// ДИНАМІКА ПРОДАЖІВ</div>', unsafe_allow_html=True)

    # Верхні метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Тижнів даних</div>
            <div class="metric-value">{len(weekly)}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Макс. продажі/тиждень</div>
            <div class="metric-value good">{weekly['Sales'].max()/1e6:.2f}M</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Середні продажі</div>
            <div class="metric-value">{weekly['Sales'].mean()/1e6:.2f}M</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Категорія</div>
            <div class="metric-value warn" style="font-size:1rem">{config.get('family','ALL')or'ALL'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Sales Over Time
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(weekly['Date'], weekly['Sales'], alpha=0.12, color='#4f8ef7')
    ax.plot(weekly['Date'], weekly['Sales'], color='#4f8ef7', linewidth=0.9, alpha=0.8, label='Тижневі продажі')
    ma4  = weekly['Sales'].rolling(4,  center=True).mean()
    ma52 = weekly['Sales'].rolling(52, center=True).mean()
    ax.plot(weekly['Date'], ma4,  color='#fbbf24', linewidth=1.5, linestyle='--', label='MA-4 (місячний)')
    ax.plot(weekly['Date'], ma52, color='#f43f5e', linewidth=2.5, label='MA-52 (річний тренд)')
    ax.set_title('Динаміка тижневих продажів (2013–2017)', fontsize=13, pad=10, color='#e8eaf0')
    ax.set_xlabel('Дата'); ax.set_ylabel('Продажі')
    ax.legend(fontsize=9, facecolor='#1a1f2e', edgecolor='#2d3348')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Сезонність
    st.markdown('<div class="section-title">// СЕЗОННІСТЬ</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        month_avg   = weekly.groupby('Month')['Sales'].mean()
        month_names = ['Січ','Лют','Бер','Кві','Тра','Чер','Лип','Сер','Вер','Жов','Лис','Гру']
        colors = ['#f43f5e' if v == month_avg.max() else '#4f8ef7' for v in month_avg.values]
        bars = ax.bar(month_avg.index, month_avg.values, color=colors, edgecolor='#0f1117', linewidth=0.5)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names, fontsize=9)
        ax.set_title('Середні продажі по місяцях', fontsize=11, color='#e8eaf0')
        ax.set_ylabel('Продажі')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        year_avg = weekly.groupby('Year')['Sales'].mean()
        colors2  = ['#4f8ef7','#6366f1','#8b5cf6','#a855f7','#ec4899'][:len(year_avg)]
        ax.bar(year_avg.index.astype(str), year_avg.values, color=colors2, edgecolor='#0f1117', linewidth=0.5)
        ax.set_title('Середні продажі по роках (тренд)', fontsize=11, color='#e8eaf0')
        ax.set_ylabel('Продажі')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Аномалії
    st.markdown('<div class="section-title">// АНОМАЛІЇ (Z-SCORE)</div>', unsafe_allow_html=True)

    threshold = st.slider('Поріг Z-score', 1.5, 3.5, 2.5, 0.1)
    z_scores  = np.abs(stats.zscore(weekly['Sales']))
    anomalies = weekly[z_scores > threshold]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(weekly['Date'], weekly['Sales'], alpha=0.10, color='#4f8ef7')
    ax.plot(weekly['Date'], weekly['Sales'], color='#4f8ef7', linewidth=0.9, label='Продажі')
    ax.scatter(anomalies['Date'], anomalies['Sales'],
               color='#f43f5e', zorder=5, s=80, marker='X',
               label=f'Аномалії (Z>{threshold}): {len(anomalies)}')
    for _, row in anomalies.iterrows():
        ax.annotate(f"{row['Date'].strftime('%b %Y')}",
                    xy=(row['Date'], row['Sales']),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=8, color='#f43f5e', ha='center')
    ax.set_title(f'Виявлення аномалій (Z-score > {threshold})', fontsize=13, color='#e8eaf0')
    ax.set_xlabel('Дата'); ax.set_ylabel('Продажі')
    ax.legend(fontsize=9, facecolor='#1a1f2e', edgecolor='#2d3348')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(f'<div class="info-box">🔴 Знайдено аномалій: <b>{len(anomalies)}</b> при порозі Z > {threshold}</div>',
                unsafe_allow_html=True)

    # Декомпозиція
    st.markdown('<div class="section-title">// ДЕКОМПОЗИЦІЯ ЧАСОВОГО РЯДУ</div>', unsafe_allow_html=True)

    ts = weekly.set_index('Date')['Sales'].copy()
    ts = ts.replace(0, np.nan).interpolate().ffill().bfill()

    try:
        decomp = seasonal_decompose(ts, model='additive', period=52)
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        pairs = [
            (decomp.observed, '#4f8ef7', 'Спостережуваний ряд'),
            (decomp.trend,    '#f43f5e', 'Тренд (MA-52)'),
            (decomp.seasonal, '#34d399', 'Сезонна складова'),
            (decomp.resid,    '#8892a4', 'Залишки (аномалії)'),
        ]
        for ax, (data, color, title) in zip(axes, pairs):
            ax.plot(data, color=color, linewidth=1)
            ax.set_title(title, fontsize=10, color='#e8eaf0', pad=4)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.grid(True, alpha=0.3)
        plt.suptitle('Декомпозиція часового ряду продажів', fontsize=13, color='#e8eaf0', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f'Декомпозиція недоступна: {e}')

# ══════════════════════════════════════════════
#  ВКЛАДКА 2: ПРОГНОЗУВАННЯ
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">// ПРОГНОЗ LSTM</div>', unsafe_allow_html=True)

    # Параметри
    col1, col2 = st.columns([1, 2])
    with col1:
        n_weeks = st.slider('🔮 Кількість тижнів для прогнозу', 1, 52, 26)
        st.markdown(f'<div class="info-box">Модель прогнозує <b>{n_weeks} тижнів</b> (~{n_weeks//4} місяців) уперед на основі останніх <b>{config.get("window_size", 8)} тижнів</b> даних.</div>',
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="margin-top:1rem">
            <div class="metric-label">Вікно моделі</div>
            <div class="metric-value">{config.get('window_size', 8)} тижнів</div>
        </div>
        <div class="metric-card" style="margin-top:0.5rem">
            <div class="metric-label">Навчальна вибірка</div>
            <div class="metric-value">{int(config.get('train_ratio',0.8)*100)}%</div>
        </div>
        <div class="metric-card" style="margin-top:0.5rem">
            <div class="metric-label">Епох навчання</div>
            <div class="metric-value">{config.get('epochs_trained', '—')}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Генерація прогнозу
        WINDOW = config.get('window_size', 8)
        ts_vals = weekly['Sales'].values.reshape(-1, 1)
        ts_sc   = scaler.transform(ts_vals)

        def multistep(model, last_seq, steps, scaler, window):
            preds, seq = [], last_seq.copy()
            for _ in range(steps):
                pred = model.predict(seq.reshape(1, window, 1), verbose=0)[0, 0]
                preds.append(pred)
                seq = np.append(seq[1:], [[pred]], axis=0)
            return scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

        last_seq  = ts_sc[-WINDOW:]
        last_date = weekly['Date'].iloc[-1]
        forecast  = multistep(model, last_seq, n_weeks, scaler, WINDOW)
        fc_dates  = pd.date_range(last_date + pd.Timedelta('7D'), periods=n_weeks, freq='W')

        # Графік
        fig, ax = plt.subplots(figsize=(12, 5))
        lookback = min(52, len(weekly))
        ax.fill_between(weekly['Date'].iloc[-lookback:], weekly['Sales'].iloc[-lookback:],
                        alpha=0.10, color='#4f8ef7')
        ax.plot(weekly['Date'].iloc[-lookback:], weekly['Sales'].iloc[-lookback:],
                color='#4f8ef7', linewidth=1.8, label='Фактичні дані')
        ax.plot(fc_dates, forecast, color='#f43f5e', linewidth=2, linestyle='--',
                marker='o', markersize=4, label=f'Прогноз LSTM (+{n_weeks} тижнів)')
        ax.fill_between(fc_dates, forecast*0.92, forecast*1.08,
                        alpha=0.15, color='#f43f5e', label='±8% діапазон')
        ax.axvline(last_date, color='#8892a4', linestyle=':', linewidth=1.5, label='Останній тиждень')
        ax.set_title(f'Multi-step прогноз LSTM на {n_weeks} тижнів уперед', fontsize=13, color='#e8eaf0')
        ax.set_xlabel('Дата'); ax.set_ylabel('Продажі')
        ax.legend(fontsize=9, facecolor='#1a1f2e', edgecolor='#2d3348')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Таблиця прогнозу
    st.markdown('<div class="section-title">// ТАБЛИЦЯ ПРОГНОЗУ</div>', unsafe_allow_html=True)
    fc_table = pd.DataFrame({
        'Дата': fc_dates.strftime('%d %b %Y'),
        'Прогноз продажів': [f'{v:,.0f}' for v in forecast],
        'Нижня межа (-8%)': [f'{v*0.92:,.0f}' for v in forecast],
        'Верхня межа (+8%)': [f'{v*1.08:,.0f}' for v in forecast],
    })
    st.dataframe(fc_table, use_container_width=True, hide_index=True)

    # Статистика прогнозу
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Середній прогноз</div>
            <div class="metric-value">{forecast.mean()/1e6:.2f}M</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Мін. прогноз</div>
            <div class="metric-value warn">{forecast.min()/1e6:.2f}M</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Макс. прогноз</div>
            <div class="metric-value good">{forecast.max()/1e6:.2f}M</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  ВКЛАДКА 3: ПОРІВНЯННЯ МОДЕЛЕЙ
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">// ПОРІВНЯННЯ МОДЕЛЕЙ</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Порівняння прогнозів трьох моделей на тестовій вибірці. Менше MAE/RMSE/MAPE — краще. Золота рамка = найкращий результат.</div>',
                unsafe_allow_html=True)

    # Графік метрик
    model_colors = {'LSTM': '#4f8ef7', 'Linear Regression': '#fbbf24', 'ARIMA(5,1,0)': '#34d399'}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, ['MAE', 'RMSE', 'MAPE (%)']):
        models = metrics_df['Модель'].tolist()
        vals   = metrics_df[metric].tolist()
        colors = [model_colors.get(m, '#8892a4') for m in models]
        bars   = ax.bar(models, vals, color=colors, edgecolor='#0f1117', linewidth=0.5, width=0.6)
        best   = np.argmin(vals)
        bars[best].set_edgecolor('#ffd700')
        bars[best].set_linewidth(2.5)
        ax.set_title(metric, fontsize=12, color='#e8eaf0', pad=8)
        ax.tick_params(axis='x', rotation=15, labelsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, vals):
            unit = '%' if metric == 'MAPE (%)' else ''
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f'{val:,.1f}{unit}', ha='center', fontsize=9, color='#e8eaf0')

    plt.suptitle('LSTM vs Linear Regression vs ARIMA', fontsize=14, color='#e8eaf0', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Легенда моделей
    col1, col2, col3 = st.columns(3)
    for col, (name, color, desc) in zip(
        [col1, col2, col3],
        [
            ('LSTM', '#4f8ef7', 'Нейромережева модель з пам\'яттю. Вловлює нелінійні залежності.'),
            ('Linear Regression', '#fbbf24', 'Лінійна регресія. Проста базова модель.'),
            ('ARIMA(5,1,0)', '#34d399', 'Класична статистична модель часових рядів.'),
        ]
    ):
        col.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid {color}; text-align:left">
            <div style="color:{color}; font-weight:600; margin-bottom:0.3rem">{name}</div>
            <div style="color:#8892a4; font-size:0.82rem">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  ВКЛАДКА 4: МЕТРИКИ
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">// МЕТРИКИ ЯКОСТІ МОДЕЛЕЙ</div>', unsafe_allow_html=True)

    # Картки метрик LSTM
    lstm_row = metrics_df[metrics_df['Модель'] == 'LSTM'].iloc[0]

    st.markdown('**🧠 LSTM — основна модель**')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value good">{lstm_row['MAE']:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value good">{lstm_row['RMSE']:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">MAPE</div>
            <div class="metric-value good">{lstm_row['MAPE (%)']:.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">R²</div>
            <div class="metric-value warn">{lstm_row['R²']:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Повна таблиця
    st.markdown('**📋 Порівняльна таблиця всіх моделей**')

    def highlight_best(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        for col in ['MAE', 'RMSE', 'MAPE (%)']:
            if col in df.columns:
                best_idx = df[col].idxmin()
                worst_idx = df[col].idxmax()
                styles.loc[best_idx, col]  = 'background-color: #1a3a2a; color: #34d399; font-weight: bold'
                styles.loc[worst_idx, col] = 'background-color: #3a1a1a; color: #f43f5e'
        if 'R²' in df.columns:
            best_idx  = df['R²'].idxmax()
            worst_idx = df['R²'].idxmin()
            styles.loc[best_idx, 'R²']  = 'background-color: #1a3a2a; color: #34d399; font-weight: bold'
            styles.loc[worst_idx, 'R²'] = 'background-color: #3a1a1a; color: #f43f5e'
        return styles

    display_df = metrics_df.set_index('Модель')
    styled = display_df.style\
        .apply(highlight_best, axis=None)\
        .format({'MAE': '{:,.2f}', 'RMSE': '{:,.2f}', 'MAPE (%)': '{:.2f}%', 'R²': '{:.4f}'})\
        .set_properties(**{'text-align': 'center', 'font-size': '13px'})
    st.dataframe(styled, use_container_width=True)

    # Пояснення метрик
    st.markdown('<div class="section-title">// ОПИС МЕТРИК</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
            <b>MAE</b> — Mean Absolute Error<br>
            Середня абсолютна похибка. Показує середнє відхилення прогнозу від реального значення.
            <br><br>
            <b>RMSE</b> — Root Mean Squared Error<br>
            Середньоквадратична похибка. Штрафує більше за великі відхилення.
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
            <b>MAPE</b> — Mean Absolute Percentage Error<br>
            Відсоткова похибка. MAPE < 10% вважається хорошим результатом для роздрібних продажів.
            <br><br>
            <b>R²</b> — Коефіцієнт детермінації<br>
            Показує частку поясненої дисперсії. Від'ємний R² вказує на високу варіативність тестових даних.
        </div>""", unsafe_allow_html=True)

    # Інфо про модель
    st.markdown('<div class="section-title">// КОНФІГУРАЦІЯ МОДЕЛІ</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    info_items = [
        ('Датасет', 'Store Sales Favorita'),
        ('Вікно', f"{config.get('window_size','—')} тижнів"),
        ('Train/Test', f"{int(config.get('train_ratio',0.8)*100)}/{int((1-config.get('train_ratio',0.8))*100)}"),
        ('Епох', str(config.get('epochs_trained','—'))),
    ]
    for col, (label, val) in zip([col1,col2,col3,col4], info_items):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:1.1rem">{val}</div>
        </div>""", unsafe_allow_html=True)

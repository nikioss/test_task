#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
import ssl
import itertools
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

from api_cals import vectorization_request, decoding_request


ssl._create_default_https_context = ssl._create_stdlib_context


df = pd.read_csv(r'C:\Users\ysx12\Documents\my-forecasting-project\src\data\data.csv')

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

print(df.head(5))


# In[2]:


json_list_df = df.to_dict(orient='records')

df_vectorized, min_val, max_val = vectorization_request(
    col_time='datetime',
    col_target="load_consumption",
    json_list_df=json_list_df
)

df_vectorized = df_vectorized.drop(columns=['Unnamed: 0'], errors='ignore')
print(df_vectorized.head(5))


# In[3]:


json_list_df = df_vectorized.to_dict(orient='records')
df_decoding = decoding_request(
    col_time='datetime',
    col_target="load_consumption",
    json_list_norm_df=json_list_df,
    min_val=min_val,
    max_val=max_val
)

print(df_decoding.head(5))


# In[4]:


df_vectorized.info()


# In[5]:


lag_acf = acf(df_vectorized['load_consumption'], nlags=100)

# Строим график ACF с Plotly
fig_acf = go.Figure()
fig_acf.add_trace(go.Scatter(x=np.arange(len(lag_acf)), y=lag_acf, mode='lines+markers', name='ACF'))
fig_acf.update_layout(
    title='ACF (Автокорреляция)',
    xaxis_title='Лаги',
    yaxis_title='Корреляция',
    template='plotly_dark'
)

# PACF
lag_pacf = pacf(df_vectorized['load_consumption'], nlags=100)

fig_pacf = go.Figure()
fig_pacf.add_trace(go.Scatter(x=np.arange(len(lag_pacf)), y=lag_pacf, mode='lines+markers', name='PACF'))
fig_pacf.update_layout(
    title='PACF (Частичная автокорреляция)',
    xaxis_title='Лаги',
    yaxis_title='Корреляция',
    template='plotly_dark'
)

fig_acf.show()
fig_pacf.show()


# Проведем проверку на выбросы

# In[6]:


# Ящик с усами
fig_box = px.box(df_vectorized, y='load_consumption')
fig_box.show()


# In[7]:


Q1 = df_vectorized['load_consumption'].quantile(0.25)
Q3 = df_vectorized['load_consumption'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
iqr_outliers = df_vectorized[(df_vectorized['load_consumption'] < lower_bound) | (df_vectorized['load_consumption'] > upper_bound)]
print(f"Найдено выбросов по IQR: {len(iqr_outliers)}")


# In[8]:


fig_iqr = go.Figure()
fig_iqr.add_trace(go.Scatter(x=df_vectorized.index, y=df_vectorized['load_consumption'], mode='markers', name='Данные'))
fig_iqr.add_trace(go.Scatter(x=iqr_outliers.index, y=iqr_outliers['load_consumption'],
                             mode='markers', marker=dict(color='red'), name='Выбросы'))
fig_iqr.update_layout(title="Выбросы по IQR", xaxis_title="Индекс", yaxis_title="Потребление")
fig_iqr.show()


# In[9]:


#список external features
exog_cols = [col for col in df_vectorized.columns if col not in ['datetime', 'load_consumption']]


# In[10]:


# ADF-тест
print(adfuller(df['load_consumption']))


# Временной ряд стационарен, т.к p-value примерно = 0. Следовательно дифференцирования не нужно(параметр d=0)

# ## Оценочный прогноз

# In[11]:


# Разделение данных
df_train = df_vectorized.iloc[:-288].reset_index(drop=True)
df_evaluate = df_vectorized.iloc[-288:].reset_index(drop=True)

exog_train = df_train[exog_cols]
exog_evaluate = df_evaluate[exog_cols]


# In[12]:


model = SARIMAX(endog=df_train['load_consumption'],
                exog=exog_train,
                order=(2, 0, 3),
                enforce_stationarity=False,
                enforce_invertibility=False)


# ```
# model = SARIMAX(endog=df_train['load_consumption'],
#                 exog=exog_train,
#                 order=(1, 0, 1),
#                 seasonal_order = (1, 1, 1, 24),
#                 enforce_stationarity=False,
#                 enforce_invertibility=False)
# ```

# ```
# best_aic = float("inf")
# best_p, best_q = 0, 0
# 
# for p in range(4):
#     for q in range(4):
#         try:
#             model = SARIMAX(df_train['load_consumption'],
#                             exog=exog_train,
#                             order=(p, 0, q),
#                             enforce_stationarity=False,
#                             enforce_invertibility=False)
#             model_fit = model.fit(disp=False)
#             
#             if model_fit.aic < best_aic:
#                 best_aic = model_fit.aic
#                 best_p, best_q = p, q
#                 best_model = model_fit
#         except:
#             pass
# 
# print(f"Лучшие параметры: p={best_p}, q={best_q}, AIC={best_aic:.2f}")
# ```

# Лучшие параметры: p=2, q=3, AIC=-484203.64

# In[13]:


model_fit = model.fit(
    disp=True)


# ```
# forecast_evaluate = best_model.predict(start=len(df_train), 
#                                        end=len(df_train) + len(df_evaluate) - 1,
#                                        exog=exog_evaluate)
# mape = mean_absolute_percentage_error(df_evaluate['load_consumption'], forecast_evaluate) * 100
# r2 = r2_score(df_evaluate['load_consumption'], forecast_evaluate)
# 
# print(f"MAPE: {mape:.2f}%")
# print(f"R²: {r2:.4f}")
# 
# MAPE: 17.13%
# R²: 0.1379
# ```

# In[14]:


forecast_evaluate = model_fit.predict(start=len(df_train), 
                                      end=len(df_train) + len(df_evaluate) - 1,
                                      exog=exog_evaluate)

# Вычисление метрик качества: MAPE и R²
mape = mean_absolute_percentage_error(df_evaluate['load_consumption'], forecast_evaluate) * 100
r2 = r2_score(df_evaluate['load_consumption'], forecast_evaluate)

print(f"Оценочный прогноз - MAPE: {mape:.2f}%")
print(f"Оценочный прогноз - R²: {r2:.4f}")


# In[15]:


real_values = df_vectorized['load_consumption'][-288:].values
exog_data = df_vectorized[exog_cols]

exog_forecast = exog_data[-288:].values

forecast_values = model_fit.forecast(steps=288, exog=exog_forecast)

# График сравнительного прогноза
fig_comparison = go.Figure()

# Добавляем реальные значения
fig_comparison.add_trace(go.Scatter(x=np.arange(len(real_values)), y=real_values, mode='lines', name='Реальные значения', line=dict(color='blue')))

# Добавляем прогнозируемые значения
fig_comparison.add_trace(go.Scatter(x=np.arange(len(forecast_values)), y=forecast_values, mode='lines', name='Прогнозируемые значения', line=dict(color='red', dash='dot')))

fig_comparison.update_layout(
    title='Сравнительный прогноз',
    xaxis_title='Точки времени',
    yaxis_title='Значение',
    template='plotly_dark'
)

fig_comparison.show()


# Метрики модели получились средне удовлетворительными. На графике сравнительного прогноза можно заметить, что модель:
# - Недостаточно учитывает экстремальные значения
# - Возможно, пропускает сезонные паттерны
# - Общая тенденция к сглаживанию ("запаздыванию") прогноза
# В качесте улучшений модели можно предложить добавление сезонности, но за неимением вычислительной мощности это гипотеза не была проверена.

# ## Реальный прогноз 

# In[16]:


df_vectorized['datetime'] = pd.to_datetime(df_vectorized['datetime'], format='%Y-%m-%dT%H:%M:%S')
last_date = df_vectorized['datetime'].iloc[-1]

freq = timedelta(minutes=5)

# Создаём будущие временные метки (288 шагов вперёд)
future_dates = [last_date + freq * (i + 1) for i in range(288)]
future_df = pd.DataFrame({'datetime': future_dates})

future_df['datetime'] = future_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
future_df['load_consumption'] = 0  # Заполняем target фиктивным значением

# Векторизация будущих дат для получения экзогенных признаков
future_json = future_df.to_dict(orient='records')
df_future_vectorized, _, _ = vectorization_request(
    col_time='datetime',
    col_target="load_consumption",
    json_list_df=future_json
)

exog_future = df_future_vectorized[exog_cols]


# In[17]:


df_vectorized = df_vectorized.drop(columns=['datetime'])
exog_data = df_vectorized.drop(columns=['load_consumption'])


# In[18]:


model_full = SARIMAX(endog=df_vectorized['load_consumption'],
                     exog=exog_data,
                     order=(2, 0, 3),
                     enforce_stationarity=False,
                     enforce_invertibility=False)
model_full_fit = model_full.fit(disp=False)


# In[19]:


real_forecast = model_full_fit.forecast(steps=288, exog=exog_future)

# График реального прогноза
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=future_dates, y=real_forecast, mode='lines', name='Реальный прогноз', line=dict(color='green')))

fig_forecast.update_layout(
    title='Реальный прогноз ',
    xaxis_title='Дата и время',
    yaxis_title='load_consumption',
    template='plotly_dark'
)

fig_forecast.show()


# В ходе работы были реализованы два типа прогнозов для временного ряда: оценочный (на исторических данных с метриками MAPE и R²) и реальный (на будущие периоды). В начале был проведен общий анализ данных, были построены графики для обнаружения выбросов, но в итоге было принято решение не избавляться от них, так как они несут в себе важную информацию. В качестве модели была выбрана SARIMAX с учетом экзогенных переменных, были подобраны оптимальные гиперпараметры p=2, d=0, q=3. Были получены метрики MAPE = 14.74% и R² = 0.3359. Необходимо учесть сезонность, так как она поможет учесть экстремальные значения и сезонные патерны, которые в данных присутсвуют.

# In[ ]:





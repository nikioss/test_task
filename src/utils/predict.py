"""
### Описание задания

1. Перед началом работы создайте форк от `main` и назовите его своим никнеймом.
2. Мы используем менеджер пакетов `pdm`. Как его установить и добавлять библиотеки — смотрите в `README`.

### Необходимо реализовать два типа прогноза временного ряда:

#### 1. Оценочный прогноз
- Создаётся на основе последних 288 точек DataFrame (`df`).
- Последние 288 точек отделяются, формируя `df_train` и `df_evaluate`.
- Модель обучается на `df_train`, затем строит прогноз на 288 точек.
- Полученные прогнозные значения сравниваются с `df_evaluate`.

#### 2. Реальный прогноз
- Выполняется на 288 точек вперёд от последней известной даты в `df`.

### Ожидаемый результат
Код должен выводить:
- График сравнительного прогноза.
- Две метрики оценки качества: `MAPE` и `R²`.
- График реального прогноза.

### Стек технологий
- **Модель:** `ARIMA` с использованием `external_features`.
- **Визуализация:** `plotly`.

### О `external_features`
`external_features` — это векторизация даты, выполняемая на нашей стороне.
Мы преобразуем строку с датой в вектор значений. Пример вызова представлен ниже. Вызов функции vectorization_request .
Эти данные уже нормализованы и должны использоваться при обучении модели.

### Дополнительно
Все материалы должны быть сохранены в директорию `results`.
"""

import ssl
import pandas as pd

from api_cals import vectorization_request, decoding_request


ssl._create_default_https_context = ssl._create_stdlib_context

df = pd.read_csv('src/data/data.csv')

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

print(df.head(5))

json_list_df = df.to_dict(orient='records')

df_vectorized, min_val, max_val = vectorization_request(
    col_time='datetime',
    col_target="load_consumption",
    json_list_df=json_list_df
)

print(df_vectorized.head(5))

json_list_df = df_vectorized.to_dict(orient='records')
df_decoding = decoding_request(
    col_time='datetime',
    col_target="load_consumption",
    json_list_norm_df=json_list_df,
    min_val=min_val,
    max_val=max_val
)

print(df_decoding.head(5))
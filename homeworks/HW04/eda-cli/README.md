# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор (`overview`)

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт (`report`)

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

#### Параметры команды `report`

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--out-dir` | Каталог для отчёта | `reports` |
| `--sep` | Разделитель в CSV | `,` |
| `--encoding` | Кодировка файла | `utf-8` |
| `--max-hist-columns` | Максимум числовых колонок для гистограмм | `6` |
| `--top-k-categories` | Сколько top-значений выводить для категориальных признаков | `5` |
| `--title` | Заголовок отчёта в `report.md` | `EDA-отчёт` |
| `--min-missing-share` | Порог доли пропусков для выделения проблемных колонок | `0.1` (10%) |

#### Примеры запуска с новыми параметрами

```bash
# Базовый вызов
uv run eda-cli report data/example.csv --out-dir reports_example

# С пользовательским заголовком и увеличенным числом категорий
uv run eda-cli report data/example.csv --out-dir reports_example --title "Анализ датасета пользователей" --top-k-categories 10

# С изменённым порогом пропусков и количеством гистограмм
uv run eda-cli report data/example.csv --out-dir reports_example --min-missing-share 0.05 --max-hist-columns 10

# Полный пример со всеми новыми параметрами
uv run eda-cli report data/example.csv --out-dir reports_custom \
    --title "Детальный EDA-анализ" \
    --top-k-categories 8 \
    --max-hist-columns 10 \
    --min-missing-share 0.05
```

## Эвристики качества данных

Функция `compute_quality_flags` в `core.py` вычисляет следующие флаги качества:

### Базовые эвристики

- `too_few_rows` — в датасете менее 100 строк
- `too_many_columns` — в датасете более 100 колонок
- `too_many_missing` — максимальная доля пропусков в какой-либо колонке > 50%

### Новые эвристики (HW03)

- `has_constant_columns` — есть колонки, где все значения одинаковые (константные)
- `has_high_cardinality_categoricals` — есть категориальные признаки с очень большим числом уникальных значений (по умолчанию порог = 50)
- `has_many_zero_values` — есть числовые колонки с долей нулей выше порога (по умолчанию 50%)

### Интегральный показатель `quality_score`

Скор качества (`quality_score`) от 0 до 1 учитывает все эвристики:
- Штраф за пропуски (пропорционально максимальной доле)
- Штраф за малое количество строк (-0.2)
- Штраф за большое количество колонок (-0.1)
- Штраф за константные колонки (-0.05 за каждую)
- Штраф за высокую кардинальность (-0.05 за каждую)
- Штраф за колонки с большой долей нулей (-0.05 за каждую)

## Тесты

```bash
uv run pytest -q
```

Тесты покрывают:
- Базовый функционал `summarize_dataset`
- Таблицу пропусков и качественные флаги
- Корреляционную матрицу и top-категории
- Новые эвристики: `has_constant_columns`, `has_high_cardinality_categoricals`, `has_many_zero_values`

---

## HTTP API (HW04)

Проект включает HTTP-сервис на базе FastAPI для оценки качества датасетов.

### Запуск сервиса

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

После запуска:
- API доступен по адресу: http://localhost:8000
- Интерактивная документация (Swagger): http://localhost:8000/docs

### Эндпоинты

#### `GET /health`

Проверка состояния сервиса.

**Ответ:**
```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}
```

#### `POST /quality`

Оценка качества по агрегированным признакам датасета.

**Запрос (JSON):**
```json
{
  "n_rows": 1000,
  "n_cols": 10,
  "max_missing_share": 0.2,
  "numeric_cols": 6,
  "categorical_cols": 4
}
```

**Ответ:**
```json
{
  "ok_for_model": true,
  "quality_score": 0.75,
  "message": "Данных достаточно, модель можно обучать (по текущим эвристикам).",
  "latency_ms": 1.23,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "no_numeric_columns": false,
    "no_categorical_columns": false
  },
  "dataset_shape": {
    "n_rows": 1000,
    "n_cols": 10
  }
}
```

#### `POST /quality-from-csv`

Оценка качества по загруженному CSV-файлу (использует EDA-ядро).

**Запрос:** multipart/form-data с CSV-файлом

**Ответ:** аналогичен `/quality`, но флаги и оценка рассчитываются на основе реального анализа CSV.

#### `POST /quality-flags-from-csv` ⭐ (новый в HW04)

Возвращает **полный набор флагов качества** из HW03 без интегрального скора.

**Запрос:** multipart/form-data с CSV-файлом

**Ответ:**
```json
{
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": true,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": true,
    "has_many_zero_values": false
  }
}
```

**Флаги качества:**
- `too_few_rows` — датасет содержит < 100 строк
- `too_many_columns` — датасет содержит > 100 колонок
- `too_many_missing` — максимальная доля пропусков > 50%
- `has_constant_columns` — есть колонки с одинаковыми значениями (HW03)
- `has_high_cardinality_categoricals` — категориальные признаки с > 50 уникальными значениями (HW03)
- `has_many_zero_values` — числовые колонки с долей нулей > 50% (HW03)

### Примеры запросов

**С использованием curl:**

```bash
# Проверка health
curl http://localhost:8000/health

# Загрузка CSV для проверки качества
curl -X POST http://localhost:8000/quality-from-csv \
  -F "file=@data/example.csv"

# Получение только флагов качества
curl -X POST http://localhost:8000/quality-flags-from-csv \
  -F "file=@data/example.csv"
```

**С использованием Python (requests):**

```python
import requests

# Отправка CSV
with open("data/example.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/quality-flags-from-csv",
        files={"file": f}
    )
    print(response.json())
```

# Smart Parking ANPR — распознавание автомобильных номеров

Итоговый проект по курсу «Инженерия Искусственного Интеллекта».

---

## 1. Паспорт проекта

- **Название проекта:** Smart Parking ANPR — сервис автоматического распознавания автомобильных номеров для систем контроля доступа (СКУД)
- **Автор:** Гутуев Владимир Евгеньевич
- **Группа:** БФБО-01-24
- **Контакт:** @Gutuevv, vgutuev@gmail.com

- **Краткое описание:**
  Проект — это REST API-сервис, который по изображению с камеры возвращает распознанный номерной знак автомобиля и уверенность модели. Решение строится как двухступенчатый пайплайн: детекция области номера (YOLOv8) и OCR (EasyOCR / PaddleOCR). Сервис задумывается как универсальный backend для разных сценариев СКУД: умный шлагбаум, автоматическая оплата парковки в ТЦ, журнал въездов/выездов на охраняемых объектах. Метрика успеха — Full Sequence Accuracy > 85% на валидационной выборке.

### Сценарии использования

1. **Умный шлагбаум на парковке ЖК / БЦ.** Камера снимает подъезжающую машину, сервис возвращает номер, СКУД сравнивает с белым списком и поднимает шлагбаум.
2. **Парковки ТЦ без билетов.** Номер распознаётся на въезде/выезде, оплата автоматически списывается с привязанной карты.
3. **Журнал въезд/выезд на охраняемых объектах.** История проездов с таймстемпами и кадрами.

---

## 2. Структура проекта

```
project/
├── README.md              # этот файл — паспорт и инструкции
├── report.md              # отчёт по экспериментам и результатам
├── self-checklist.md      # чеклист самопроверки перед сдачей
├── requirements.txt       # зависимости проекта
├── Dockerfile             # сборка контейнера сервиса
├── .dockerignore
├── src/                   # исходный код
│   ├── config.py          # настройки через pydantic-settings
│   ├── pipeline.py        # склейка detector + OCR
│   ├── models/
│   │   ├── detector.py    # обёртка над YOLOv8 (stub + реальная загрузка)
│   │   └── ocr.py         # обёртка над EasyOCR
│   └── service/
│       ├── app.py         # FastAPI приложение
│       └── schemas.py     # Pydantic схемы запросов/ответов
├── configs/
│   ├── .env.example       # шаблон переменных окружения
│   └── config.yaml        # параметры пайплайна
├── data/                  # демо-картинки и небольшие выборки
├── notebooks/             # EDA, эксперименты
├── tests/                 # smoke + unit-тесты
└── artifacts/             # сохранённые веса моделей и метрики
```

---

## 3. Требования и установка

### 3.1. Требования

- Python `>= 3.10` (рекомендуется 3.11)
- На Linux/WSL: пакет `libgl1` для opencv-python-headless (обычно уже стоит).
- Для GPU-инференса — CUDA-совместимая видеокарта (опционально; CPU тоже поддерживается).

### 3.2. Локальная установка (uv / venv)

```powershell
# Из корня репозитория
cd project

# Вариант A: стандартный venv
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

pip install --upgrade pip
pip install -r requirements.txt
```

```powershell
# Вариант B: uv (быстрее)
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3.3. Конфигурация

Скопируйте шаблон переменных окружения и при необходимости отредактируйте:

```powershell
copy configs\.env.example .env
```

Ключевые параметры:

| Переменная        | По умолчанию | Назначение                                            |
|-------------------|--------------|--------------------------------------------------------|
| `HOST`            | `0.0.0.0`    | Адрес сервиса                                         |
| `PORT`            | `8000`       | Порт сервиса                                          |
| `OCR_LANGS`       | `en,ru`      | Языки EasyOCR (через запятую)                         |
| `USE_DETECTOR`    | `false`      | Если `true`, перед OCR применяется YOLOv8-детектор    |
| `DETECTOR_WEIGHTS`| `yolov8n.pt` | Путь до весов детектора                               |
| `MIN_CONFIDENCE`  | `0.3`        | Порог уверенности для отбрасывания результатов        |

---

## 4. Как запустить

### 4.1. Запуск сервиса локально

```powershell
cd project
.venv\Scripts\activate
python -m uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --reload
```

После запуска откройте Swagger UI: http://localhost:8000/docs

### 4.2. Эндпоинты

- `GET  /health` — health-check, возвращает `{"status": "ok"}`.
- `GET  /metrics` — счётчики запросов и ошибок (in-memory).
- `POST /predict` — принимает изображение (multipart `file`), возвращает:

  ```json
  {
    "plate": "A123BC77",
    "confidence": 0.91,
    "detections": [
      {"bbox": [x1, y1, x2, y2], "text": "A123BC77", "confidence": 0.91}
    ]
  }
  ```

### 4.3. Пример запроса

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -H "accept: application/json" `
  -F "file=@data/sample.jpg"
```

### 4.4. Запуск в Docker

```powershell
cd project
docker build -t smart-parking-anpr .
docker run --rm -p 8000:8000 --env-file .env smart-parking-anpr
```

---

## 5. Данные

- **CCPD (Chinese City Parking Dataset)** — для обучения и валидации YOLOv8-детектора (большой объём, лежит вне репозитория, инструкции в `data/README.md`).
- **Nomeroff Net datasets** — для проверки распознавания, в том числе на номерах русского формата.
- В репозитории хранятся только небольшие демо-картинки в `data/` для smoke-проверки сервиса. Большие датасеты не коммитятся.

---

## 6. Тесты

```powershell
cd project
.venv\Scripts\activate
pytest tests
```

Минимальный smoke-тест проверяет, что приложение поднимается и `/health` отвечает 200.

---

## 7. Демонстрация на защите

1. Поднять сервис: `python -m uvicorn src.service.app:app --reload`.
2. Открыть Swagger UI и через `/predict` подать 2-3 кадра с парковки.
3. Показать в `report.md` сравнение baseline (только OCR) vs YOLO+OCR по Full Sequence Accuracy.
4. Кратко пройтись по структуре `src/` и пайплайну `pipeline.py`.

---

## 8. Ограничения и дальнейшая работа

Текущие ограничения:

- В базовой версии детектор отключён (`USE_DETECTOR=false`) — OCR работает по всему кадру; качество падает на мелких номерах.
- Кириллица в EasyOCR требует донастройки whitelist символов — сделано через параметр `OCR_LANGS` и пост-фильтр.
- Качество чувствительно к освещению, грязи и ракурсу.
- `/metrics` пока in-memory (сбрасывается при рестарте); Prometheus exporter — TODO.

Дальше планируется:

- Дообучение YOLOv8n на CCPD и включение детектора по умолчанию.
- Добавление пост-обработки: regex-валидация формата номера по стране.
- Базовая авторизация по API-ключу и логирование запросов в файл/Loki.
- Prometheus exporter и трекинг экспериментов через MLflow.

---

## 9. Источники, лицензии и атрибуция

В соответствии с правилами курса (`COURSE-POLICIES.md` §6) фиксируем источники данных и кода:

**Данные:**

- [CCPD (Chinese City Parking Dataset)](https://github.com/detectRecog/CCPD) — открытый датасет автомобильных номеров Китая. Используется для обучения детектора YOLOv8.
- [Nomeroff Net datasets](https://github.com/ria-com/nomeroff-net) — открытые наборы для валидации OCR, в том числе номера российского формата.

**Модели и библиотеки:**

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — детектор; используются предобученные веса `yolov8n.pt` и собственный fine-tuning.
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — оптическое распознавание текста.
- [FastAPI](https://fastapi.tiangolo.com/) — REST-сервис.

В репозитории не хранятся секреты и реальные персональные данные — только демо-кадры и шаблон `.env.example`.

---

## 10. Соответствие требованиям курса

Проект следует методическим документам мета-репозитория [`mirea-aie-2025/aie-course-meta`](https://github.com/mirea-aie-2025/aie-course-meta):

- `project/overview.md` §4 — пройден набор этапов 1, 8–13; этапы 2–7 (EDA, baseline-сравнения, трекинг экспериментов, выбор финальной модели) — в работе.
- `evaluation/project-evaluation.md` — самооценка зафиксирована в `self-checklist.md`.
- `COURSE-POLICIES.md` §4.2 — выполнены пункты 1, 4, 5, 6, 7; пункты 2, 3 (полный data-pipeline и серия экспериментов) — на следующих итерациях.

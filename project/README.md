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
│   ├── pipeline.py        # склейка detector + OCR (выбор движка)
│   ├── postprocess.py     # нормализация номера + RU-постобработка
│   ├── evaluate.py        # оценка EasyOCR (FSA/CER, ablation)
│   ├── train_ocr.py       # обучение CRNN (CTC)
│   ├── train_detector.py  # обучение YOLOv8-детектора
│   ├── eval_crnn.py       # оценка CRNN на test
│   ├── eval_nomeroff.py   # оценка Nomeroff (reference)
│   ├── benchmark.py       # бенчмарк эффективности CRNN
│   ├── benchmark_nomeroff.py
│   ├── data/
│   │   ├── nomeroff.py    # загрузчик OCR-датасета
│   │   ├── ocr_dataset.py # torch Dataset + кодек символов
│   │   └── prepare_detector.py  # HF-датасет → YOLO-формат
│   ├── models/
│   │   ├── detector.py    # обёртка над YOLOv8 (stub + реальная загрузка)
│   │   ├── ocr.py         # обёртка над EasyOCR (baseline)
│   │   ├── crnn.py        # CRNN-модель (CNN+BiLSTM+CTC)
│   │   └── crnn_ocr.py    # инференс-движок CRNN для сервиса
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
| `OCR_ENGINE`      | `crnn`       | Движок OCR: `crnn` (обученная модель) или `easyocr`   |
| `CRNN_WEIGHTS`    | `artifacts/crnn.pt` | Веса CRNN (`crnn.pt` точность, `crnn_tiny.pt` edge) |
| `OCR_LANGS`       | `en,ru`      | Языки EasyOCR (только при `OCR_ENGINE=easyocr`)        |
| `USE_DETECTOR`    | `false`      | Если `true`, перед OCR применяется YOLOv8-детектор    |
| `DETECTOR_WEIGHTS`| `yolov8n.pt` | Путь до весов детектора                               |
| `MIN_CONFIDENCE`  | `0.3`        | Порог уверенности (EasyOCR)                            |
| `USE_GPU`         | `false`      | Инференс на GPU                                        |

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
# Демо-кроп номера (ground truth виден в имени файла)
curl.exe -X POST "http://localhost:8000/predict" `
  -H "accept: application/json" `
  -F "file=@data/samples/crops/A001BP54.png"
```

Демо-картинки лежат в `data/samples/` (кропы для CRNN и полные кадры для режима детектора).

### 4.4. Запуск в Docker

```powershell
cd project
docker build -t smart-parking-anpr .
docker run --rm -p 8000:8000 --env-file .env smart-parking-anpr
```

### 4.5. Обучение и оценка моделей

```powershell
# Обучение CRNN (полная модель)
python -m src.train_ocr --data-dir <dataset> --epochs 15 --batch-size 256 --gpu --out artifacts/crnn.pt

# Лёгкий вариант для edge (1.95M параметров)
python -m src.train_ocr --data-dir <dataset> --gpu --last-channels 256 --rnn-hidden 128 --rnn-layers 1 --out artifacts/crnn_tiny.pt

# Оценка на test (FSA/CER)
python -m src.eval_crnn --data-dir <dataset> --split test --weights artifacts/crnn.pt --gpu
python -m src.evaluate  --data-dir <dataset> --split test --gpu        # EasyOCR baseline (ablation)

# Бенчмарк эффективности
python -m src.benchmark --weights artifacts/crnn.pt --gpu
```

### 4.6. Детектор номера (полный кадр)

```powershell
# Подготовка датасета детекции в YOLO-формат (нужен отдельный env с datasets, см. report §7)
python -m src.data.prepare_detector --out <path>/plates_yolo

# Обучение YOLOv8n (test mAP@50 = 0.99)
python -m src.train_detector --data <path>/plates_yolo/data.yaml --epochs 40 --device 0
```

Чтобы `/predict` работал на **полном кадре** (а не на кропе номера), включите детектор:
`OCR_ENGINE=crnn`, `USE_DETECTOR=true` — тогда пайплайн сам находит номер и читает его.

---

## 5. Данные

- **AUTO.RIA Numberplate OCR RU** (Nomeroff Net) — основной датасет: кропы РФ-номеров с разметкой
  (train 49 382 / val 4 893 / test 2 845). Скачивается с https://nomeroff.net.ua/datasets/.
- **keremberke/license-plate-object-detection** (HuggingFace) — для обучения YOLOv8-детектора
  (~8.8k размеченных кадров → YOLO-формат train 6176 / val 1765 / test 882).
- В репозитории хранятся обученные веса (`artifacts/crnn.pt`, `crnn_tiny.pt`,
  `artifacts/detector/plate_yolov8n.pt`) и результаты оценки (`artifacts/*.csv`); большие
  датасеты не коммитятся (инструкции в `data/README.md`).

## 5a. Результаты

Сравнение на едином test-сплите (2845 номеров), подробности — в [`report.md`](./report.md):

| Модель | test FSA | Параметры | Размер |
|---|---|---|---|
| EasyOCR + постобработка (baseline) | 22.0% | — | — |
| **CRNN-full (финальная)** | **99.54%** | 5.31M | 21 МБ |
| CRNN-tiny (edge) | 98.66% | 1.95M | 7.8 МБ |
| Nomeroff Net (reference) | 99.58% | 3.47M | ~14 МБ |

Наша CRNN на уровне production-референса по точности; tiny-вариант превосходит его по компактности
и single-image latency. Цель проекта (FSA > 85%) перекрыта.

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

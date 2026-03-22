# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- **Часть A:** выбран датасет STL10 (10 классов, 96x96 RGB, 5000 train / 8000 test) — хороший баланс между сложностью и размером. Проведены 4 эксперимента (C1–C4): простая CNN, CNN с аугментациями, ResNet18 head-only и ResNet18 fine-tune.
- **Часть B:** выбран датасет OxfordIIITPet и трек `segmentation`. Использована pretrained-модель DeepLabV3_ResNet50 в двух режимах постобработки (V1 — baseline argmax, V2 — удаление мелких компонент).
- В части A сравнивались: эффект аугментаций (C1 vs C2) и transfer learning (C3/C4 vs C1/C2). В части B сравнивались два режима постобработки по mean IoU, pixel precision и pixel recall.

## 2. Среда и воспроизводимость

- Python: 3.10
- torch / torchvision: 2.x / 0.x
- Устройство: CUDA (T4 GPU, Google Colab)
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: **STL10**
- Разделение: train 4000 / val 1000 / test 8000 (train/val split 80/20 с seed=42)
- Базовые transforms: `ToTensor → Normalize(STL10_mean, STL10_std)`
- Augmentation transforms: `RandomHorizontalFlip(0.5) + RandomCrop(96, padding=8) + ColorJitter(0.2, 0.2, 0.2) → ToTensor → Normalize`
- Комментарий: STL10 содержит 10 классов (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck) с изображениями 96x96. Малый размер train-выборки (5000) делает задачу нетривиальной для простых CNN и создаёт хороший кейс для демонстрации преимуществ transfer learning. Изображения достаточно крупные (96x96 vs 32x32 у CIFAR), чтобы ResNet18 с ресайзом до 224x224 мог эффективно использовать pretrained-фичи.

### 3.2. Часть B: structured vision

- Датасет: **OxfordIIITPet**
- Трек: `segmentation`
- Что считается ground truth: бинарная маска «питомец» (trimap == 1)
- Какие предсказания использовались: DeepLabV3 argmax по классам `cat` (idx 8) и `dog` (idx 12) → бинарная маска
- Комментарий: OxfordIIITPet содержит ~3700 trainval-изображений кошек и собак с аннотациями тримапов. Бинарная постановка «питомец vs фон» разумна, т.к. pretrained DeepLabV3 на COCO-VOC знает оба класса (cat, dog), а ground truth тримапы содержат чёткую разметку foreground. Это позволяет корректно оценить качество сегментации через IoU.

## 4. Часть A: модели и обучение (C1-C4)

- **C1 (simple-cnn-base):** SimpleCNN (3 Conv блока + AdaptiveAvgPool + 2 Linear, 620 362 параметра), без аугментаций, 96x96 input.
- **C2 (simple-cnn-aug):** та же SimpleCNN, с аугментациями (HFlip + RandomCrop + ColorJitter).
- **C3 (resnet18-head-only):** ResNet18 pretrained ImageNet, backbone заморожен, обучается только fc-голова (5 130 trainable params), 224x224 input.
- **C4 (resnet18-finetune):** ResNet18 pretrained ImageNet, размораживаем layer4 + fc (8 398 858 trainable params), backbone lr=1e-4, head lr=1e-3.

Дополнительно:

- Loss: CrossEntropyLoss
- Optimizer(ы): Adam (lr=1e-3 для CNN и head; для C4 — lr=1e-4 backbone, lr=1e-3 head, weight_decay=1e-4)
- Batch size: 64
- Epochs (макс): 15 (C1, C2), 10 (C3, C4)
- Критерий выбора лучшей модели: максимальная `val_accuracy` за все эпохи

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Segmentation track

- Модель: DeepLabV3_ResNet50 (COCO_WITH_VOC_LABELS_V1), только инференс
- Что считается foreground: пиксели, где argmax модели = `cat` (8) или `dog` (12)
- V1: базовая постобработка — argmax → бинарная маска (cat|dog = 1, остальное = 0)
- V2: альтернативная постобработка — то же + удаление связных компонент < 500 пикселей (`skimage.morphology.remove_small_objects`)
- Как считался mean IoU: per-image binary IoU (intersection / union для pet vs background), затем среднее по 150 изображениям
- Считались ли дополнительные pixel-level метрики: да, pixel precision и pixel recall

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации сегментации: `./artifacts/figures/segmentation_examples.png`
- Метрики сегментации: `./artifacts/figures/segmentation_metrics.png`

Короткая сводка:

- Лучший эксперимент части A: **C4 (resnet18-finetune)**
- Лучшая `val_accuracy`: **0.948**
- Итоговая `test_accuracy` лучшего классификатора: **0.9449**
- Что дали аугментации (C2 vs C1): небольшой прирост val accuracy с 0.598 до 0.613 (+1.5 п.п.), аугментации помогают, но не кардинально при малом количестве данных и простой архитектуре.
- Что дал transfer learning (C3/C4 vs C1/C2): колоссальный скачок — с ~0.6 до ~0.95. Pretrained ResNet18 радикально эффективнее простой CNN на STL10.
- Что оказалось лучше: head-only или partial fine-tuning: fine-tuning (C4: 0.948) немного лучше head-only (C3: 0.942), но разница невелика (+0.6 п.п.).
- Что показал режим V1 во второй части: mean IoU = 0.7279, precision = 0.7355, recall = 0.988 — модель находит почти все пиксели питомца (высокий recall), но захватывает лишние области фона (precision ниже).
- Что показал режим V2 во второй части: mean IoU = 0.7281, precision = 0.7357, recall = 0.988 — удаление мелких компонент практически не повлияло на метрики, основные ошибки — крупные области, а не мелкий шум.
- Как интерпретируются метрики второй части: IoU — строгая метрика, штрафующая и за ложные срабатывания, и за пропуски. Высокий recall при среднем precision говорит о том, что модель скорее «пересегментирует», чем пропускает объект.

## 7. Анализ

Простая CNN (C1, C2) на STL10 достигает val accuracy около 0.6 — это ожидаемо для мелкой модели с 620K параметрами на 4000 тренировочных изображениях. Архитектура из трёх свёрточных блоков слишком поверхностна, чтобы извлечь сложные иерархические признаки, необходимые для различения 10 классов. Аугментации (C2) дают умеренное улучшение (+1.5 п.п.), что логично: они регуляризируют модель и увеличивают разнообразие данных, но не компенсируют ограниченную ёмкость простой CNN.

Transfer learning радикально меняет ситуацию: pretrained ResNet18 даже в режиме head-only (C3) достигает val accuracy 0.942. Это объясняется тем, что ImageNet-фичи (текстуры, формы, паттерны) хорошо переносятся на STL10, который по сути является подмножеством ImageNet-подобных изображений. Partial fine-tuning (C4: 0.948) даёт дополнительный, но скромный прирост — layer4 адаптируется к специфике STL10, однако большая часть полезных признаков уже заложена в замороженных слоях.

В части B (сегментация) IoU ~0.73 при recall ~0.99 показывает, что DeepLabV3 успешно находит питомцев, но захватывает лишние фоновые области. Переход от V1 к V2 (удаление мелких компонент) практически не повлиял на метрики, что говорит о том, что основные ошибки — это крупные области неправильной классификации (например, фон рядом с питомцем), а не мелкий шум. Наиболее показательные ошибки — случаи, когда модель путает фоновую мебель или одежду с питомцем, а также пропускает тонкие конечности (уши, хвост).

## 8. Итоговый вывод

В качестве базового конфига классификации для STL10 мы бы выбрали C4 (ResNet18 finetune layer4+fc): он даёт лучший баланс качества (test accuracy 0.9449) и разумной сложности обучения. Transfer learning — ключевой инструмент при ограниченных данных: даже без fine-tuning (head-only) pretrained ResNet18 превосходит обученную с нуля CNN в 1.5 раза по accuracy. Для задачи сегментации pretrained DeepLabV3 даёт приемлемое качество out-of-the-box (mean IoU ~0.73), но для дальнейшего улучшения потребовалось бы fine-tuning на целевом датасете, а простая постобработка (удаление шума) не решает основных проблем с границами объектов.

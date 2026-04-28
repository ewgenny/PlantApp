# 🌿 LeafPulse

LeafPulse — десктопное приложение на базе Vision Transformer (ViT) для автоматического определения типа растения и распознавания заболеваний томатов и огурцов по изображениям листьев.

Программа анализирует загруженное фото, определяет культуру растения, затем выбирает соответствующую модель и выявляет возможное заболевание. Результат выводится в графическом интерфейсе вместе с уровнем уверенности, а также сохраняется в историю диагностик с возможностью последующего экспорта данных.

---

## ⚙️ Возможности

- 🔄 Двухэтапная классификация:
  - ConvNeXt-Tiny — определение культуры (томат / огурец)
  - ViT-B/16 — распознавание заболеваний:
    - Томаты: 10 классов (9 болезней + healthy)
    - Огурцы: 7 классов
- 🖼 Поддержка выбора изображения и drag-and-drop
- 🎨 Цветовая индикация результата
- 📖 Описание заболевания и рекомендации по лечению
- 🗂 История диагностик (SQLite)
- 📤 Экспорт в CSV и TXT
- 💻 Полностью оффлайн-работа

---

## 🧠 Использованные технологии

- Python
- PyTorch
- timm (Vision Transformer, ConvNeXt)
- Torchvision
- PyQt5
- SQLite
- Pillow (PIL)
- NumPy
- scikit-learn
- tqdm

---

## 🛠 Установка и настройка проекта

### Создание папки проекта

```bash
mkdir plantApp
cd plantApp
```

### Создание виртуального окружения

```bash
python -m venv venv
```

Активация:

```bash
venv\Scripts\activate
```

### Установка зависимостей

```bash
pip install torch torchvision timm pyqt5 pillow tqdm scikit-learn matplotlib pandas
```

### Структура проекта

```text
plantApp/
│── data/
│── models/
│── logs/
│── src/
│   │── gui.py
│   │── predict.py
│   │── train_vit_tomato.py
│   │── train_vit_cucumber.py
│   │── train_binary.py
│   │── preprocess.py
│   │── data_loader.py
│   │── disease_info.py
```

### 📊 Подготовка датасетов

Используются открытые датасеты:

#### Томаты (10 классов)

- PlantVillage Tomato Dataset — [скачать с Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Или альтернативная ссылка: [Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)

#### Огурцы (7 классов)

- Cucumber Disease Dataset — [скачать с Kaggle](https://www.kaggle.com/datasets/shubhambatra98/cucumber-disease)

Культуры растений:

- Tomato — 10 классов (9 болезней + healthy)
- Cucumber — 7 классов

Структура данных:

```text
data/tomato/
    train/
    val/
    test/

data/cucumber/
    train/
    val/
    test/
```

Каждый класс хранится в отдельной папке:

```text
train/
   bacterial_spot/
   early_blight/
   healthy/
```

### 📥 Загрузка данных

Используется DataLoader:

train_loader, val_loader, test_loader, class_names = get_dataloaders(...)

### 🌳 Обучение бинарной модели

Модель: ConvNeXt-Tiny
Задача: определение культуры растения

```bash
python src/train_binary.py
```

Сохранение:

```bash
models/binary_convnext_best.pth
```

### 🍅 Обучение модели томатов

Модель: ViT-B/16
Классы: 10

```bash
python src/train_vit_tomato.py
```

Сохранение:

```bash
models/vit_tomato_best.pth
```

### 🥒 Обучение модели огурцов

```bash
python src/train_vit_cucumber.py
```

Сохранение:

```bash
models/vit_cucumber_best.pth
```

### 🔍 Логика работы модели

```text
Фото листа
   ↓
ConvNeXt-Tiny (binary)
   ↓
tomato / cucumber
   ↓
ViT (соответствующая модель)
   ↓
класс болезни + confidence
```

### 🖥 Графический интерфейс

Реализован с использованием библиотеки PyQt5 и включает следующие элементы интерфейса:

- отображение логотипа приложения в главном окне;
- загрузка изображения через кнопку выбора файла;
- поддержка перетаскивания изображения в окно приложения (drag-and-drop);
- кнопка запуска диагностики;
- отображение загруженного изображения;
- вывод результата классификации (тип растения, заболевание, уровень уверенности);
- отображение описания заболевания и рекомендаций;
- просмотр истории предыдущих диагностик;
- экспорт сохранённых результатов в CSV и TXT.

### Итоговая структура программы:

```text
PLANTAPP/
├── .vscode/                       # настройки Visual Studio Code и конфигурация проекта
├── data/                          # датасеты для обучения и тестирования моделей
│   ├── binary/                    # изображения для бинарной классификации (томат / огурец)
│   ├── cucumber/                  # датасет заболеваний огурцов
│   └── tomato/                    # датасет заболеваний томатов
├── logs/                          # файлы истории работы приложения
│   ├── diagnoses.db               # база данных SQLite с историей диагностик
│   └── history.csv                # экспортированные результаты в CSV
├── models/                        # обученные модели нейросетей
│   ├── binary_convnext_best.pth   # модель ConvNeXt-Tiny для определения культуры
│   ├── vit_cucumber_best.pth      # ViT-модель для диагностики болезней огурцов
│   └── vit_tomato_best.pth        # ViT-модель для диагностики болезней томатов
├── src/                           # исходный код проекта
│   ├── __pycache__/               # автоматически созданные Python-кэш файлы
│   ├── data_loader.py             # загрузка датасетов через DataLoader
│   ├── database.py                # работа с SQLite: сохранение истории и запросы
│   ├── disease_info.py            # описание болезней и рекомендации по лечению
│   ├── evaluate_binary.py         # тестирование бинарной модели
│   ├── evaluate_vit_cucumber.py   # оценка точности модели огурцов
│   ├── evaluate_vit_tomato.py     # оценка точности модели томатов
│   ├── predict.py                 # логика предсказания по изображению
│   ├── preprocess.py              # предобработка изображений (resize, normalize)
│   ├── train_binary.py            # обучение бинарного классификатора
│   ├── train_vit_cucumber.py      # обучение ViT-модели для огурцов
│   ├── train_vit_tomato.py        # обучение ViT-модели для томатов
│   └── utils.py                   # вспомогательные функции проекта
├── venv/                          # виртуальное окружение Python
└── gui.py                         # главный файл графического интерфейса PyQt5
```

### 🗄 Хранение данных (SQLite)

Для сохранения истории диагностик используется база данных SQLite.

Файл базы данных:

logs/diagnoses.db

В базе данных сохраняется следующая информация:

- дата и время проведения диагностики;
- определённый тип растения;
- выявленное заболевание;
- уровень уверенности модели (%);
- путь к загруженному изображению.

### 📤 Экспорт данных

Приложение поддерживает экспорт истории диагностик в следующие форматы:

CSV — для последующего анализа в Excel или других табличных редакторах;
TXT — для текстового хранения результатов.

### 🚀 Запуск проекта

Для запуска графического интерфейса приложения необходимо выполнить команду:

```bash
python gui.py
```

## Точность моделей

Бинарный классификатор (ConvNeXt-Tiny) → ~98–100%
(на простых датасетах вроде PlantVillage при чистых данных может доходить до 100%, но обычно 98–99.5% из-за шума и аугментаций)
ViT-B/16 (томаты, 10 классов) → ~95–99%
(Vision Transformer стабильно показывает очень высокую точность на PlantVillage; до ~99% при хорошем обучении)
ViT-B/16 (огурцы, 7 классов) → ~97–99.5%
(для cucumber datasets ViT часто даёт ~99% accuracy при нормальной выборке и балансировке классов)

## 👥 Автор

- **Деркач Евгения Владимировна** — разработка и обучение моделей, GUI и интеграция
- GitHub: [ewgenny](https://github.com/ewgenny)
- Email: egenderkach20@gmail.com

## 📄 Лицензия

Этот проект распространяется под лицензией **MIT License**.

MIT License

Copyright (c) 2025 LeafPulse

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# 🌸 Классификация изображений цветов с SigLip

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

## 📋 Оглавление
- [О проекте](#-о-проекте)
- [Датасет](#-датасет)
- [Результаты](#-результаты)
- [Эксперименты](#-эксперименты)
- [Установка и запуск](#-установка-и-запуск)
- [Мониторинг](#-мониторинг)

## 🎯 О проекте

Проект посвящен решению задачи классификации изображений цветов с использованием современного подхода **SigLip**. Основные цели:
- Реализация точной классификации 102 видов цветов
- Исследование эффективности метода SigLip
- Оптимизация процесса обучения модели

## 📊 Датасет

**Flowers102** - высококачественный датасет для классификации цветов.

### Характеристики:
- **Классы**: 102 вида цветов
- **Объем данных**: 8,189 изображений
  - 🔵 Тренировочная выборка: 6,149
  - 🟡 Валидационная выборка: 1,020
  - 🔴 Тестовая выборка: 1,020
- **Формат**: RGB изображения различного разрешения

> 🔗 [Официальная страница датасета](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

## 📈 Результаты

### Метрики на тестовой выборке

| Метрика | Значение |
|---------|----------|
| Accuracy | 96.4% |
| Precision | 96.6% |
| Recall | 96.4% |
| F1-Score | 96.3% |

В датасете 102 класса, precision, recall, accuracy, f1 - для каждого из классов можно найти в ./metrics/full_test_metrics.json


### Визуализация обучения

<div align="center">
  <img src="./metrics/train_loss.jpg" width="45%" alt="Loss График"/>
  <img src="./metrics/accuracy.jpg" width="45%" alt="Accuracy График"/>
  <img src="./metrics/f1_score.jpg" width="45%" alt="F1 score График"/>
  <img src="./metrics/precision.jpg" width="45%" alt="Precision График"/>
  <img src="./metrics/recall.jpg" width="45%" alt="Recall График"/>
</div>

## 🧪 Эксперименты

**Ключевые наблюдения:**
- Модель переобучилась на трейн и результаты на тестовой выборке были ужасны

## 🚀 Установка и запуск

```bash
# Клонирование репозитория
git clone <repository-url>
cd <repository-name>

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 📊 Мониторинг

### AIM
```bash
aim up
```
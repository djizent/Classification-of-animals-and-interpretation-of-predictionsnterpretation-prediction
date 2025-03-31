# Инструкция по развертыванию Streamlit-приложения для классификации животных

## Подготовка модели и приложения
Убедитесь, что у вас есть следующие файлы:
- Модель: `models/efficientnet_b0_animals.pth`
- Список классов: `class_names.txt`
- Код приложения: `app.py` (содержит предоставленный Streamlit-код)

Создайте файл `requirements.txt` со следующим содержимым:
```text
streamlit
torch
torchvision
lime
Pillow
matplotlib
numpy
scikit-image
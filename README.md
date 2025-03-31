# Инструкция по развертыванию Streamlit-приложения для классификации животных

# Инструкция по обучению модели классификации животных

## 1. Подготовка окружения
Для обучения модели использовались следующие библиотеки:
```python
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from torch import nn, optim
from tqdm import tqdm
```
## 2. Загрузка и подготовка данных
Использовался датасет "mertcobanov/animals" с Hugging Face:

```
dataset = load_dataset("mertcobanov/animals")
```
### Преобразования изображений:
```
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```
### Создание кастомного Dataset:
```
class AnimalsDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset["train"]
        self.transform = transform
        self.labels = list(set(self.data["label"]))
        self.label_map = {label: i for i, label in enumerate(self.labels)}
    
    # ... методы __len__ и __getitem__ ...
```
## 3. Разделение данных
Данные разделены на три части:

- 80% - тренировочный набор

- 10% - валидационный набор

- 10% - тестовый набор

```
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
```
## 4. Создание модели
Использована предобученная EfficientNet-B0 с заменой последнего слоя:

```
model = models.efficientnet_b0(pretrained=True)
num_classes = len(dataset.labels)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
```
## 5. Обучение модели
Параметры обучения:
- Устройство: CUDA если доступно, иначе CPU

- Функция потерь: CrossEntropyLoss

- Оптимизатор: Adam с learning rate 0.001

- Количество эпох: 15

- Размер батча: 32

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
### Процесс обучения:
```
for epoch in range(epochs):
    # ... тренировочный цикл с tqdm прогресс-баром ...
    val_acc = calculate_accuracy(val_loader)
    print(f"Epoch {epoch+1} completed. Avg Loss: {running_loss / len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
```
## 6. Сохранение результатов
После обучения:

- Сохранены веса модели: efficientnet_b0_animals_15.pth

- Сохранена полная модель: efficientnet_b0_animals_full.pth

- Создан файл с названиями классов: class_names.txt

```
torch.save(model.state_dict(), "efficientnet_b0_animals_15.pth")
torch.save(model, "efficientnet_b0_animals_full.pth")

with open("class_names.txt", "w", encoding="utf-8") as file:
    for label in class_names:
        file.write(f"{label}\n")
```
## 7. Тестирование модели
Финальная точность на тестовом наборе:

```
test_acc = calculate_accuracy(test_loader)
print(f"Финальная точность на тесте: {test_acc:.4f}")
```
## 8. Пример предсказания
Функция для тестирования на отдельных изображениях:

```
def predict_from_test(index):
    # ... загрузка изображения и предсказание ...
    print(f"Реальный класс: {real_label_text}")
    print(f"Предсказанный класс: {predicted_label_text}")
    # ... отображение изображения с подписями ...
```
## Ссылки
Датасет: mertcobanov/animals на Hugging Face
https://huggingface.co/datasets/mertcobanov/animals

Использованная архитектура: EfficientNet-B0




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
```
## Создание Docker-образа

Создайте Dockerfile со следующим содержимым:
```
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
## Соберите Docker-образ:
```
docker build -t animal-classifier .
```
## Проверка работы локально

Запустите контейнер локально:
```
docker run -p 8501:8501 animal-classifier
```
После запуска приложение будет доступно по адресу: http://localhost:8501

## Публикация образа в Docker Hub

1. Авторизуйтесь в Docker Hub:
```
docker login
```
2. Создайте тег для вашего образа:
```
docker tag animal-classifier ваш_логин_на_dockerhub/animal-classifier
```
3. Загрузите образ в Docker Hub:
```
docker push ваш_логин_на_dockerhub/animal-classifier
```
## Развертывание на виртуальной машине

1. Создайте виртуальную машину (например, в Яндекс.Облаке):

    - ОС: Ubuntu 20.04/22.04 LTS

    - Минимальные характеристики: 2 vCPU, 2GB RAM, 10GB SSD

    - Добавьте SSH-ключ для доступа

2. Подключитесь к виртуальной машине:
```
ssh -l username <IP-адрес ВМ>
```
3. Установите Docker:
```
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```
4. Запустите контейнер с вашим приложением:
```
docker run -d -p 8501:8501 --restart unless-stopped ваш_логин_на_dockerhub/animal-classifier
```
Приложение будет доступно по адресу: http://<IP-адрес ВМ>:8501

## Использование приложения

1. Откройте веб-интерфейс по указанному адресу

2. Загрузите изображение животного через интерфейс

3. Приложение покажет:

    - Топ-5 наиболее вероятных классов животных

    - Визуализацию LIME, объясняющую, на какие части изображения обратила внимание модель

## Дополнительные настройки

Для повышения производительности вы можете:

    - Увеличить ресурсы виртуальной машины

    - Использовать GPU-ускорение (требует установки nvidia-docker)

    - Настроить балансировку нагрузки, если ожидается высокий трафик
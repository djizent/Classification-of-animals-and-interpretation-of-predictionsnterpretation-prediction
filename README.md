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

sДля повышения производительности вы можете:

    - Увеличить ресурсы виртуальной машины

    - Использовать GPU-ускорение (требует установки nvidia-docker)

    - Настроить балансировку нагрузки, если ожидается высокий трафик
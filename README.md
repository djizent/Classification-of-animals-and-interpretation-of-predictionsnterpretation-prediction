Animal Classification with Streamlit and Docker
Это приложение классифицирует изображения животных с помощью предобученной модели EfficientNet и визуализирует объяснения предсказаний через LIME.

Требования
Docker (установите с официального сайта)

Docker Compose (обычно идет вместе с Docker)

Сервер с Linux (Ubuntu 20.04/22.04 рекомендуется)

Установка и запуск
1. Клонируйте репозиторий
bash
Copy
git clone <ваш-репозиторий>
cd <папка-проекта>
2. Соберите Docker-образ
bash
Copy
docker build -t animal-classification .
3. Запустите контейнер
Вариант 1: Обычный запуск
bash
Copy
docker run -p 8501:8501 animal-classification
Вариант 2: Запуск в фоновом режиме
bash
Copy
docker run -d -p 8501:8501 --name animal-app animal-classification
Вариант 3: С Docker Compose (рекомендуется)
Создайте файл docker-compose.yml:

yaml
Copy
version: '3.8'

services:
  app:
    image: animal-classification
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./class_names.txt:/app/class_names.txt
    restart: unless-stopped
Запустите:

bash
Copy
docker-compose up -d
4. Откройте приложение
Приложение будет доступно по адресу:

Copy
http://<IP-адрес-сервера>:8501
Структура проекта
Copy
.
├── Dockerfile
├── README.md
├── app.py              # Основной скрипт Streamlit
├── models/
│   └── efficientnet_b0_animals.pth  # Модель PyTorch
├── class_names.txt     # Список классов
└── requirements.txt    # Зависимости Python
Dockerfile
dockerfile
Copy
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
requirements.txt
Copy
streamlit==1.32.0
torch==2.1.0
torchvision==0.16.0
lime==0.2.0.1
Pillow==10.1.0
numpy==1.26.0
matplotlib==3.8.0
scikit-image==0.22.0
Настройка сервера
Убедитесь, что порт 8501 открыт в фаерволе:

bash
Copy
sudo ufw allow 8501
Для постоянной работы используйте systemd (создайте /etc/systemd/system/animal-app.service):

ini
Copy
[Unit]
Description=Animal Classification App
After=docker.service

[Service]
Restart=always
ExecStart=/usr/bin/docker-compose -f /path/to/your/project/docker-compose.yml up
ExecStop=/usr/bin/docker-compose -f /path/to/your/project/docker-compose.yml down

[Install]
WantedBy=multi-user.target
Затем:

bash
Copy
sudo systemctl daemon-reload
sudo systemctl enable animal-app
sudo systemctl start animal-app
Обновление приложения
Остановите контейнер:

bash
Copy
docker-compose down
Обновите код и пересоберите образ:

bash
Copy
docker-compose build --no-cache
Запустите снова:

bash
Copy
docker-compose up -d
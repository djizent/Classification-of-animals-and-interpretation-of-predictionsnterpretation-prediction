FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8051

CMD ["streamlit", "run", "--server.port", "8051", "st_app.py"]
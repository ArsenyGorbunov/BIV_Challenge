# Используем официальный образ Python 3.11.10
FROM python:3.11.10-slim

# Устанавливаем переменные окружения для более предсказуемого поведения Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Устанавливаем ограничения по памяти и процессорам
ENV MEMORY=8g \
    CPU_LIMIT=4

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . /app

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Команда запуска при старте контейнера
CMD ["python", "predict.py"]

# 1. Используем официальный образ Python
FROM python:3.11-slim

# 2. Устанавливаем ffmpeg (он нужен для обработки аудио Muxlisa)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 3. Создаем рабочую директорию
WORKDIR /app

# 4. Копируем и устанавливаем зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Копируем код нашего бота
COPY bot.py .

# 6. Команда для запуска бота
CMD ["python", "bot.py"]
# ── Базовий образ ─────────────────────────────────────────────────────────────
# python:3.11-slim — Debian-based, mediapipe офіційно підтримує 3.8–3.12.
# Python 3.13 (який Railway підбирав раніше) не підтримується mediapipe.
FROM python:3.11-slim

# ── Системні залежності для OpenCV headless ───────────────────────────────────
# opencv-python-headless потребує libgl1 + libglib2.0-0.
# libxcb1 / libx11 НЕ потрібні — ми явно використовуємо headless-збірку.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Залежності (окремим шаром — кешується при незмінному requirements.txt) ────
COPY requirements.txt .

# Крок 1: встановлюємо всі залежності зі списку.
# Крок 2: примусово перевстановлюємо headless-версію OpenCV.
#   mediapipe тягне opencv-python (з GUI, потребує X11/libxcb) як transitive dep,
#   тому --force-reinstall після основного install гарантує, що cv2 буде headless.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall "opencv-python-headless>=4.8.0"

# ── Код застосунку ────────────────────────────────────────────────────────────
COPY . .

# ── Запуск ────────────────────────────────────────────────────────────────────
# Railway передає порт через змінну $PORT; fallback = 8000 для локального запуску.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

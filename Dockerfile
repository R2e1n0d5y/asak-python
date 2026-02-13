# Gunakan image Python 3.9 yang ramping
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Instal dependensi sistem untuk OpenCV
# Menggunakan libgl1 dan libglib2.0-0 sebagai pengganti paket lama
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements dulu
COPY requirements.txt .

# Instal library python
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode aplikasi
COPY . .

# Ekspos port Flask
EXPOSE 5000

# Ganti port ke 7860 agar dikenali Hugging Face
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app", "--timeout", "120"]
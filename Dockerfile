# Gunakan image Python 3.9 yang ramping
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Instal dependensi sistem untuk OpenCV dan pendukung lainnya
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements dulu agar bisa memanfaatkan cache Docker
COPY requirements.txt .

# Instal library python
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode aplikasi dan folder model/assets ke dalam container
COPY . .

# Ekspos port yang digunakan Flask (default 5000)
EXPOSE 5000

# Jalankan aplikasi menggunakan Gunicorn untuk stabilitas produksi
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
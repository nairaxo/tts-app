FROM python:3.10-slim

# Dépendances système minimales
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Création du répertoire de travail
# WORKDIR /app

# Copie des fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY ./app /app

# Exposition du port
EXPOSE 8000

# Lancement de l’application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

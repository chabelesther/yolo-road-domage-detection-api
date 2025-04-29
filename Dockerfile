# Utiliser une image de base Python
FROM python:3.9-slim

# Installer les dépendances système nécessaires pour OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Définir les variables d'environnement pour matplotlib et ultralytics
ENV MPLCONFIGDIR=/tmp/matplotlib_cache
ENV YOLO_CONFIG_DIR=/tmp/ultralytics_cache

# Créer un utilisateur non-root pour exécuter l'application
RUN useradd -m appuser

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY . .

# Donner les permissions à l'utilisateur non-root
RUN chown -R appuser:appuser /app

# Passer à l'utilisateur non-root
USER appuser

# Exposer le port
EXPOSE 8000

# Ajouter une vérification de santé
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
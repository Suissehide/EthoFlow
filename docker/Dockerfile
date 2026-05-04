# Image Docker pour les TESTS LOCAUX du pipeline (Streamlit + scripts utilitaires).
#
# Cette image contient uniquement la couche orchestration / UI.
# Elle n'embarque PAS DeepLabCut ni VAME — ces deux frameworks ont besoin
# d'une GPU NVIDIA et d'envs complexes, et tournent en natif sur le poste
# de calcul Windows.
#
# Build :
#   docker build -t pipeline-souris -f docker/Dockerfile .
# Run :
#   docker run -p 8501:8501 -v $(pwd)/data:/app/data pipeline-souris
# Ou via compose :
#   docker compose -f docker/docker-compose.yml up

FROM python:3.10-slim

WORKDIR /app

# Dépendances système : ffmpeg pour le crop, libGL pour OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

# Code de l'app
COPY streamlit_app/ ./streamlit_app/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]

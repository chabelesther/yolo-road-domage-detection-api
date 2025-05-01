from ultralytics import YOLO
import cv2
import time
import numpy as np
import threading
import queue
from flask import Flask, Response, render_template_string
import logging
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration globale
CONFIG = {
    "CAMERA_INDEX": 1,  # Index de la webcam (0 pour la première, 1 pour la seconde, etc.)
    "FRAME_WIDTH": 640,  # Résolution de la frame
    "FRAME_HEIGHT": 480,
    "JPEG_QUALITY": 80,  # Qualité JPEG (0-100)
    "QUEUE_SIZE": 10,    # Taille maximale de la queue
}

# Variables globales
frame_queue = queue.Queue(maxsize=CONFIG["QUEUE_SIZE"])
running = True
model = None

# Charger le modèle YOLO
try:
    model = YOLO("best.pt")
    logger.info("Modèle YOLO chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle YOLO : {e}")
    exit(1)

# Fonction pour capturer et traiter les frames de la webcam
def process_webcam():
    global running
    cap = None

    try:
        # Initialiser la webcam
        cap = cv2.VideoCapture(CONFIG["CAMERA_INDEX"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["FRAME_WIDTH"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])

        if not cap.isOpened():
            logger.error("Erreur : impossible d'ouvrir la webcam")
            running = False
            return

        logger.info("Webcam activée. Traitement en cours...")

        prev_time = time.time()
        while running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.warning("Erreur lors de la capture de la frame")
                time.sleep(0.1)
                continue

            # Calculer les FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time > prev_time else 0
            prev_time = current_time

            try:
                # Inférence avec YOLO
                results = model(frame)
                annotated_frame = results[0].plot()

                # Ajouter les FPS
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Ajouter la frame à la queue
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(annotated_frame)
            except Exception as e:
                logger.error(f"Erreur lors du traitement de la frame : {e}")
                continue

    except Exception as e:
        logger.error(f"Erreur dans process_webcam : {e}")
    finally:
        if cap is not None:
            cap.release()
        logger.info("Webcam désactivée")

# Fonction pour encoder les frames en JPEG
def get_frame():
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                # Encoder en JPEG avec qualité configurable
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG["JPEG_QUALITY"]]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"Erreur lors de l'encodage de la frame : {e}")
        else:
            time.sleep(0.01)

# Créer l'application Flask
app = Flask(__name__)

# Page d'accueil
@app.route('/')
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Détection de nids-de-poule - Webcam</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f4f4f4;
                text-align: center;
            }
            h1 { color: #333; }
            .video-container {
                margin: 20px auto;
                max-width: 90%;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
                overflow: hidden;
            }
            .video-stream { width: 100%; height: auto; display: block; }
            .status {
                margin-top: 20px;
                padding: 10px;
                background-color: #e7f7e7;
                border-radius: 5px;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <h1>Détection en temps réel via webcam</h1>
        <div class="video-container">
            <img src="/video_feed" class="video-stream" alt="Flux vidéo">
        </div>
        <div class="status">
            Webcam active - Traitement en cours
        </div>
    </body>
    </html>
    """
    return render_template_string(html_content)

# Route pour le flux vidéo
@app.route('/video_feed')
def video_feed():
    return Response(
        get_frame(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# Fonction principale
if __name__ == '__main__':
    processing_thread = None
    try:
        # Lancer le thread de traitement webcam
        processing_thread = threading.Thread(target=process_webcam)
        processing_thread.daemon = True
        processing_thread.start()

        logger.info("Démarrage du serveur web. Accédez à http://127.0.0.1:5000/")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Arrêt du serveur...")
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")
    finally:
        running = False
        if processing_thread and processing_thread.is_alive():
            processing_thread.join(timeout=1.0)
        logger.info("Streaming terminé")
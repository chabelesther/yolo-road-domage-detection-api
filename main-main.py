from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import logging
import asyncio
import time
import os
import sys
import threading
 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable globale pour le modèle
model = None
model_lock = threading.Lock()

# Fonction pour charger le modèle à la demande
def get_model():
    global model
    with model_lock:
        if model is None:
            try:
                model = YOLO("best.pt")
            except Exception as e:
                print(f"Erreur lors du chargement du modèle: {str(e)}")
                raise e
    return model

@app.get("/")
async def root():
    # Affichons les informations sur les versions
    import ultralytics
    return {
        "greeting": "Hello, World!",
        "message": "Welcome to FastAPI!",
        "ultralytics_version": ultralytics.__version__,
        "python_version": sys.version,
        "model_path": os.path.abspath("best.pt"),
        "model_exists": os.path.exists("best.pt"),
        "working_directory": os.getcwd()
    }

# Précharger le modèle au démarrage pour éviter la latence du premier appel
@app.on_event("startup")
async def startup_event():
    try:
        get_model()
        logger.info("Modèle YOLO préchargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du préchargement du modèle: {str(e)}")

 
@app.websocket("/ws/stream-video")
async def stream_video(websocket: WebSocket):
    await websocket.accept()
    frames_processed = 0
    timeout_count = 0
    last_frame_time = time.time()
    skip_frames = 0
    
    try:
        model = get_model()
        logger.info("Connexion WebSocket établie pour le streaming vidéo")
        await websocket.send_json({"status": "ready"})

        while True:
            # Réception des chunks vidéo avec timeout
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)
                timeout_count = 0  # Réinitialiser le compteur de timeouts
                
                data_size = len(data) if data else 0
                
                if not data or data_size < 1000:
                    logger.warning(f"Données insuffisantes reçues: {data_size} bytes")
                    continue
                
                # Convertir les données binaires en array numpy
                nparr = np.frombuffer(data, np.uint8)
                
                # Vérifier si c'est une image JPEG valide
                if nparr[0] != 0xFF or nparr[1] != 0xD8:
                    logger.warning("Format non-JPEG détecté, tentative de décodage quand même")
                
                # Décodage de l'image
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning("Échec du décodage de l'image")
                    await websocket.send_json({"error": "Échec du décodage de l'image"})
                    continue
                
                # Contrôle adaptatif du débit en fonction du temps de traitement
                current_time = time.time()
                processing_time = current_time - last_frame_time
                
                # Si le traitement est trop lent, on saute des images
                if processing_time < 0.05:  # Temps de traitement très rapide
                    skip_frames = 0
                elif processing_time > 0.2:  # Traitement lent
                    skip_frames = min(skip_frames + 1, 2)  # Augmenter progressivement, max 2
                
                if skip_frames > 0 and frames_processed % (skip_frames + 1) != 0:
                    # On saute cette frame pour réduire la charge
                    frames_processed += 1
                    continue
                
                # Récupérer les dimensions originales
                original_height, original_width = frame.shape[:2]
                
                # Conversion en RGB pour YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Prédiction YOLO avec conf plus élevée pour moins de détections
                results = model.predict(frame_rgb, conf=0.45, verbose=False)
                
                # Obtenir le résultat annoté mais conserver les dimensions originales
                annotated_frame = results[0].plot()
                
                # Compression et envoi avec qualité adaptée pour réduire la latence
                # Qualité plus faible pour améliorer la latence tout en conservant les dimensions
                _, buffer_jpeg = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                frame_base64 = base64.b64encode(buffer_jpeg).decode("utf-8")
                
                # Calculer FPS
                current_time = time.time()
                fps = 1.0 / (current_time - last_frame_time) if current_time > last_frame_time else 0
                last_frame_time = current_time
                
                # Envoi au frontend
                await websocket.send_json({
                    "frame": frame_base64,
                    "timestamp": current_time,
                    "fps": round(fps, 1),
                    "dimensions": {"width": original_width, "height": original_height}
                })
                
                frames_processed += 1
                if frames_processed % 10 == 0:
                    logger.info(f"Frame #{frames_processed} traitée, FPS: {round(fps, 1)}, Skip: {skip_frames}")
                
            except asyncio.TimeoutError:
                timeout_count += 1
                logger.warning(f"Timeout #{timeout_count} lors de la réception")
                try:
                    await websocket.send_json({"ping": time.time()})
                except:
                    logger.error("Impossible d'envoyer le ping, connexion probablement fermée")
                    break
                if timeout_count > 3:  # Réduit de 5 à 3 pour une détection plus rapide des déconnexions
                    logger.error("Trop de timeouts consécutifs, fermeture de la connexion")
                    break
                continue
            except Exception as e:
                logger.error(f"Erreur lors de la réception: {str(e)}")
                if "connection closed" in str(e).lower():
                    break
                continue

    except WebSocketDisconnect:
        logger.info("Client déconnecté")
    except Exception as e:
        logger.error(f"Erreur serveur: {str(e)}")
        try:
            await websocket.send_json({"error": f"Erreur serveur: {str(e)}"})
        except:
            pass
    finally:
        await websocket.close()
        logger.info(f"Connexion WebSocket fermée. Total frames traitées: {frames_processed}")
 
if __name__ == "__main__":
    import uvicorn
    
    # Récupérer le port depuis les variables d'environnement, avec une valeur par défaut
    port = int(os.environ.get("PORT", 8000))
    
    # Log de démarrage
    logger.info(f"Démarrage du serveur sur le port {port}")
    
    # Démarrer le serveur avec des paramètres optimisés
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
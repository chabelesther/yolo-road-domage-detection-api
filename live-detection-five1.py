from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import av  # Pour décoder les chunks WebM en mémoire
from io import BytesIO
import logging
import asyncio

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configurer CORS pour permettre les connexions depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable globale pour le modèle YOLO
model = None

# Fonction pour charger le modèle YOLO à la demande
def get_model():
    global model
    if model is None:
        try:
            logger.info("Chargement du modèle YOLO...")
            model = YOLO("best.pt")  # Assurez-vous que best.pt est dans le répertoire _

            logger.info("Modèle YOLO chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise e
    return model

# Endpoint de santé pour vérifier que l'API est opérationnelle
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Endpoint WebSocket pour le streaming vidéo
@app.websocket("/ws/stream-video")
async def stream_video(websocket: WebSocket):
    await websocket.accept()
    buffer = BytesIO()  # Buffer pour accumuler les chunks WebM
    try:
        # Charger le modèle YOLO
        model = get_model()
        
        logger.info("Connexion WebSocket établie pour le streaming vidéo")

        while True:
            # Recevoir un chunk vidéo (WebM)
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout lors de la réception du chunk")
                continue

            if not data:
                logger.warning("Chunk vide reçu")
                continue
            
            # Ajouter le chunk au buffer
            buffer.write(data)
            buffer.seek(0)

            # Tenter de décoder le buffer avec pyav
            try:
                container = av.open(buffer, mode="r", format="webm")
                for frame in container.decode(video=0):
                    # Convertir la frame en numpy array (RGB)
                    frame_rgb = frame.to_ndarray(format="rgb24")
                    
                    # Effectuer la prédiction avec YOLO
                    results = model.predict(frame_rgb, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    # Convertir la frame annotée en JPEG pour réduire la bande passante
                    _, buffer_jpeg = cv2.imencode(".jpg", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    frame_base64 = base64.b64encode(buffer_jpeg).decode("utf-8")
                    
                    # Envoyer la frame annotée au frontend
                    await websocket.send_json({"frame": frame_base64})
                    logger.debug("Frame annotée envoyée au frontend")
                
                container.close()
            except av.error.InvalidDataError as e:
                logger.warning(f"Données WebM invalides, accumulation des chunks: {str(e)}")
                # Réinitialiser la position du buffer pour continuer à accumuler
                buffer.seek(0, 2)  # Aller à la fin du buffer
                continue
            except Exception as e:
                logger.error(f"Erreur lors du traitement de la frame: {str(e)}")
                await websocket.send_json({"error": f"Erreur lors du traitement de la frame: {str(e)}"})
                buffer.seek(0, 2)  # Continuer à accumuler
                continue
            
            # Réinitialiser le buffer après un décodage réussi
            buffer = BytesIO()
            
    except WebSocketDisconnect:
        logger.info("Client déconnecté")
    except Exception as e:
        logger.error(f"Erreur serveur: {str(e)}")
        await websocket.send_json({"error": f"Erreur serveur: {str(e)}"})
    finally:
        # Fermer le buffer
        buffer.close()
        await websocket.close()
        logger.info("Connexion WebSocket fermée")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
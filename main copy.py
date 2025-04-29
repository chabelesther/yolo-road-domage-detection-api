from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from io import BytesIO
import cv2
import numpy as np
import tempfile
import os
import base64
import time

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Créer une variable globale pour le modèle
model = None

# Fonction pour charger le modèle à la demande
def get_model():
    global model
    if model is None:
        try:
            model = YOLO("best.pt")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            raise e
    return model

# Endpoint de santé pour les healthchecks
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Obtenir le modèle
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du modèle: {str(e)}")
    
    # Vérifier le type de fichier
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "video/mp4"]:
        raise HTTPException(status_code=400, detail="Type de fichier non pris en charge")

    # Lire le fichier
    contents = await file.read()

    if content_type.startswith("image/"):
        # Traitement d'image
        from PIL import Image
        image = Image.open(BytesIO(contents))
        results = model.predict(image)
        annotated_img = results[0].plot()
        img_byte_array = BytesIO()
        img = Image.fromarray(annotated_img)
        img.save(img_byte_array, format="PNG")
        img_byte_array.seek(0)
        return StreamingResponse(img_byte_array, media_type="image/png")

    elif content_type.startswith("video/"):
        # Pour une vidéo, on renvoie un message indiquant que le traitement se fera via WebSocket
        return {"message": "Vidéo reçue. Connectez-vous au WebSocket '/ws/process-video' pour recevoir les frames annotées."}

@app.websocket("/ws/process-video")
async def process_video(websocket: WebSocket):
    await websocket.accept()
    try:
        # Obtenir le modèle
        try:
            model = get_model()
        except Exception as e:
            await websocket.send_json({"error": f"Erreur lors du chargement du modèle: {str(e)}"})
            await websocket.close()
            return
            
        # Recevoir la vidéo via WebSocket
        video_data = await websocket.receive_bytes()

        # Sauvegarder temporairement la vidéo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_data)
            temp_file_path = temp_file.name

        # Lire la vidéo avec OpenCV
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            await websocket.send_json({"error": "Impossible d'ouvrir la vidéo"})
            os.unlink(temp_file_path)
            await websocket.close()
            return

        # Obtenir les propriétés de la vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # Une frame par seconde

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Traiter une frame par seconde
            if frame_count % frame_interval != 0:
                continue

            # Convertir le frame en format compatible avec YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(frame_rgb, verbose=False)
            annotated_frame = results[0].plot()

            # Convertir le frame annoté en bytes (format PNG)
            _, buffer = cv2.imencode(".png", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            frame_bytes = buffer.tobytes()

            # Encoder en base64 pour l'envoyer via WebSocket
            frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")

            # Envoyer le frame au frontend
            await websocket.send_json({
                "frame": frame_base64,
                "frame_number": frame_count
            })

            # Attendre un peu pour simuler le rythme (1 frame par seconde)
            # time.sleep(1)

        # Fin du traitement
        await websocket.send_json({"message": "Traitement terminé"})
        cap.release()
        os.unlink(temp_file_path)

    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
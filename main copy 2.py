from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from io import BytesIO
import cv2
import numpy as np
import tempfile
import os
import uuid
import shutil

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
        # Traitement de vidéo complète et renvoi de la vidéo annotée
        input_file = None
        output_file = None
        
        try:
            # Créer des fichiers temporaires pour l'entrée et la sortie
            input_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            input_file.write(contents)
            input_file.close()
            
            # Créer un fichier temporaire pour la vidéo de sortie
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output_file.close()
            
            # Ouvrir la vidéo avec OpenCV
            cap = cv2.VideoCapture(input_file.name)
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Impossible d'ouvrir la vidéo")
            
            # Obtenir les propriétés de la vidéo
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Créer un objet VideoWriter pour la vidéo de sortie
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))
            
            # Traiter et annoter chaque frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convertir le frame pour YOLO et prédire
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(frame_rgb, verbose=False)
                
                # Obtenir le frame annoté
                annotated_frame = results[0].plot()
                
                # Convertir en BGR pour OpenCV
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Écrire le frame dans la vidéo de sortie
                out.write(annotated_frame_bgr)
            
            # Libérer les ressources
            cap.release()
            out.release()
            
            # Créer une copie temporaire persistante pour FileResponse
            response_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            response_file.close()
            
            # Copier le fichier de sortie vers le fichier de réponse
            shutil.copy2(output_file.name, response_file.name)
            
            # Retourner la vidéo annotée
            return FileResponse(
                path=response_file.name,
                media_type="video/mp4",
                filename="video_annotated.mp4"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de la vidéo: {str(e)}")
        finally:
            # Nettoyer les fichiers temporaires
            if input_file and os.path.exists(input_file.name):
                os.unlink(input_file.name)
            if output_file and os.path.exists(output_file.name):
                os.unlink(output_file.name)
            # Ne pas supprimer response_file car il est utilisé par FileResponse

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
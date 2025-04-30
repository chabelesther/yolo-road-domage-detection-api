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
import glob
from io import BytesIO
import tempfile
import shutil

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

model = None

@app.get("/model-info")
async def model_info():
    """Endpoint pour obtenir des informations sur le modèle et diagnostiquer les problèmes"""
    model_path = "best.pt"
    try:
        # Informations sur le fichier
        file_info = {}
        if os.path.exists(model_path):
            file_info["exists"] = True
            file_info["size"] = os.path.getsize(model_path)
            file_info["readable"] = os.access(model_path, os.R_OK)
            file_info["path"] = os.path.abspath(model_path)
        else:
            file_info["exists"] = False
            
        # Informations environnement
        import torch
        import ultralytics
        env_info = {
            "python_version": sys.version,
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available() if hasattr(torch, 'cuda') else False,
            "working_directory": os.getcwd(),
        }
        
        return {
            "status": "ok",
            "file_info": file_info,
            "env_info": env_info
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'obtention des informations sur le modèle: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


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
@app.websocket("/ws/stream-video")
async def stream_video(websocket: WebSocket):
    await websocket.accept()
    frames_processed = 0
    timeout_count = 0
    last_frame_time = time.time()
    
    try:
        model = get_model()
        logger.info("Connexion WebSocket établie pour le streaming vidéo")
        await websocket.send_json({"status": "ready"})

        while True:
            # Réception des chunks vidéo avec timeout plus long
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)
                timeout_count = 0  # Réinitialiser le compteur de timeouts
                
                # Log détaillé de la réception
                data_size = len(data) if data else 0
                logger.info(f"Données reçues: {data_size} bytes")
                
                if not data or data_size < 1000:
                    logger.warning(f"Données insuffisantes reçues: {data_size} bytes")
                    continue
                
                # Convertir les données binaires en array numpy
                nparr = np.frombuffer(data, np.uint8)
                
                # Vérifier si c'est une image JPEG valide
                if nparr[0] == 0xFF and nparr[1] == 0xD8:
                    logger.debug("Format JPEG détecté")
                else:
                    logger.warning("Format non-JPEG détecté, tentative de décodage quand même")
                
                # Décodage de l'image
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning("Échec du décodage de l'image")
                    # Envoi d'un message d'erreur au client
                    await websocket.send_json({"error": "Échec du décodage de l'image"})
                    continue
                
                # Log des dimensions pour debug
                h, w = frame.shape[:2]
                logger.info(f"Image décodée: {w}x{h}px")
                
                # Conversion en RGB pour YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Prédiction YOLO
                results = model.predict(frame_rgb, verbose=False)
                annotated_frame = results[0].plot()
                
                # Compression et envoi
                _, buffer_jpeg = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                frame_base64 = base64.b64encode(buffer_jpeg).decode("utf-8")
                
                # Calculer le temps écoulé depuis la dernière frame
                current_time = time.time()
                fps = 1.0 / (current_time - last_frame_time) if current_time > last_frame_time else 0
                last_frame_time = current_time
                
                # Envoi au frontend
                await websocket.send_json({
                    "frame": frame_base64,
                    "timestamp": current_time,
                    "fps": round(fps, 1)
                })
                
                frames_processed += 1
                if frames_processed % 5 == 0:
                    logger.info(f"Frame #{frames_processed} traitée et envoyée, FPS: {round(fps, 1)}")
                
            except asyncio.TimeoutError:
                timeout_count += 1
                logger.warning(f"Timeout #{timeout_count} lors de la réception")
                # Envoyer un ping pour maintenir la connexion
                try:
                    await websocket.send_json({"ping": time.time()})
                except:
                    logger.error("Impossible d'envoyer le ping, connexion probablement fermée")
                    break
                # Si trop de timeouts consécutifs, on considère la connexion comme perdue
                if timeout_count > 5:
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


 
@app.get("/list-files")
async def list_files():
    """Liste tous les fichiers dans le répertoire de travail et les sous-répertoires"""
    try:
        # Répertoire courant
        current_dir = os.getcwd()
        logger.info(f"Répertoire de travail actuel: {current_dir}")
        
        # Liste des fichiers dans le répertoire courant
        files_in_root = os.listdir(current_dir)
        logger.info(f"Fichiers à la racine: {files_in_root}")
        
        # Recherche récursive des fichiers .pt
        pt_files = []
        for root, dirs, files in os.walk(current_dir):
            for file in files:
                if file.endswith('.pt'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, current_dir)
                    size = os.path.getsize(full_path)
                    pt_files.append({
                        "name": file,
                        "path": rel_path,
                        "full_path": full_path,
                        "size": size,
                        "readable": os.access(full_path, os.R_OK)
                    })
        
        # Recherche de toutes les images dans le répertoire
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']
        image_files = []
        for ext in image_extensions:
            for file_path in glob.glob(f"{current_dir}/**/{ext}", recursive=True):
                rel_path = os.path.relpath(file_path, current_dir)
                image_files.append(rel_path)
        
        return {
            "working_directory": current_dir,
            "files_in_root": files_in_root,
            "pt_files": pt_files,
            "image_files": image_files[:20]  # Limiter à 20 images pour éviter une réponse trop grande
        }
    except Exception as e:
        logger.error(f"Erreur lors du listage des fichiers: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/file-info/{file_path:path}")
async def file_info(file_path: str):
    """Obtenir des informations détaillées sur un fichier spécifique"""
    try:
        # Valider et normaliser le chemin
        if '..' in file_path:  # Empêcher la traversée de répertoire
            raise HTTPException(status_code=400, detail="Chemin non autorisé")
            
        full_path = os.path.join(os.getcwd(), file_path)
        
        if not os.path.exists(full_path):
            return JSONResponse(
                status_code=404,
                content={"error": f"Fichier non trouvé: {file_path}"}
            )
            
        # Collecter des informations sur le fichier
        stat_info = os.stat(full_path)
        file_info = {
            "name": os.path.basename(full_path),
            "path": file_path,
            "full_path": full_path,
            "size": stat_info.st_size,
            "created": stat_info.st_ctime,
            "modified": stat_info.st_mtime,
            "readable": os.access(full_path, os.R_OK),
            "writable": os.access(full_path, os.W_OK),
            "executable": os.access(full_path, os.X_OK)
        }
        
        # Si c'est un fichier .pt, essayer de charger pour tester
        if full_path.endswith('.pt'):
            try:
                # Juste tester si le fichier peut être chargé
                test_model = YOLO(full_path)
                file_info["model_loadable"] = True
            except Exception as e:
                file_info["model_loadable"] = False
                file_info["load_error"] = str(e)
        
        return file_info
    except Exception as e:
        logger.error(f"Erreur lors de l'obtention des informations sur le fichier: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    
    # Récupérer le port depuis les variables d'environnement, avec une valeur par défaut
    port = int(os.environ.get("PORT", 8000))
    
    # Log de démarrage
    logger.info(f"Démarrage du serveur sur le port {port}")
    
    # Démarrer le serveur
    uvicorn.run(app, host="0.0.0.0", port=port)
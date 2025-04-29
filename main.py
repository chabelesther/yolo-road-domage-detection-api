from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import logging
import asyncio
import time
import os
import sys

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

def get_model():
    global model
    if model is None:
        try:
            # Vérifier si le fichier existe
            model_path = "best.pt"
            if not os.path.exists(model_path):
                logger.error(f"Le fichier modèle {model_path} n'existe pas")
                # Essayons de télécharger un modèle de secours
                try:
                    logger.info("Téléchargement d'un modèle par défaut...")
                    model = YOLO("yolov8n.pt")  # Modèle par défaut à télécharger automatiquement
                    logger.info("Modèle par défaut chargé avec succès")
                    return model
                except Exception as backup_error:
                    logger.error(f"Échec du téléchargement du modèle par défaut: {str(backup_error)}")
                    raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
            
            # Vérifier les permissions
            if not os.access(model_path, os.R_OK):
                logger.error(f"Pas de permission de lecture pour {model_path}")
                raise PermissionError(f"Impossible de lire le modèle: {model_path}")
            
            # Afficher la taille du fichier
            file_size = os.path.getsize(model_path)
            logger.info(f"Taille du fichier modèle: {file_size} bytes")
            
            # Essayer de charger le modèle avec gestion d'erreur détaillée
            logger.info("Chargement du modèle YOLO...")
            try:
                model = YOLO(model_path)
                logger.info("Modèle YOLO chargé avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
                
                # Fichier probablement corrompu, tentons avec un modèle par défaut
                try:
                    logger.info("Tentative de chargement d'un modèle par défaut...")
                    # Renommer le fichier corrompu pour le sauvegarder
                    backup_path = f"{model_path}.backup"
                    if os.path.exists(model_path):
                        os.rename(model_path, backup_path)
                        logger.info(f"Fichier corrompu sauvegardé en {backup_path}")
                    
                    # Charger un modèle par défaut
                    model = YOLO("yolov8n.pt")  # Télécharge automatiquement
                    logger.info("Modèle par défaut chargé avec succès")
                except Exception as backup_error:
                    logger.error(f"Échec du chargement du modèle par défaut: {str(backup_error)}")
                    raise e
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise e
    return model

@app.post("/detect")
async def detect_image(image: UploadFile = File(...)):
    """
    Endpoint REST pour analyser une image et retourner l'image annotée.
    """
    start_time = time.time()
    
    try:
        # Vérifier le type de fichier
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")
        
        # Lire l'image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Impossible de décoder l'image")
        
        # Obtenir les dimensions de l'image
        h, w = img.shape[:2]
        logger.info(f"Image reçue: {w}x{h}px, taille: {len(contents)/1024:.1f}KB")
        
        # Conversion en RGB pour YOLO
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Charger le modèle
        model = get_model()
        
        # Effectuer la prédiction
        results = model.predict(img_rgb, verbose=False)
        annotated_img = results[0].plot()
        
        # Convertir l'image annotée en base64
        _, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Calculer le temps d'exécution
        processing_time = time.time() - start_time
        logger.info(f"Image traitée en {processing_time:.3f} secondes")
        
        # Retourner l'image annotée
        return {
            "success": True,
            "image": img_str,
            "processing_time": processing_time,
            "detections": len(results[0].boxes)
        }
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

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

@app.post("/upload-model")
async def upload_model(model_file: UploadFile = File(...)):
    """Endpoint pour télécharger un modèle personnalisé"""
    global model
    try:
        # Vérifier le type de fichier
        if not model_file.filename.endswith('.pt'):
            raise HTTPException(status_code=400, detail="Le fichier doit être un modèle PyTorch (.pt)")
        
        # Sauvegarder le fichier
        model_path = "best.pt"
        
        # Sauvegarder le modèle existant s'il y en a un
        if os.path.exists(model_path):
            backup_path = f"{model_path}.old"
            os.rename(model_path, backup_path)
            logger.info(f"Modèle existant sauvegardé en {backup_path}")
        
        # Écrire le nouveau fichier
        contents = await model_file.read()
        with open(model_path, "wb") as f:
            f.write(contents)
        
        file_size = os.path.getsize(model_path)
        logger.info(f"Modèle téléchargé: {model_file.filename}, taille: {file_size} bytes")
        
        # Réinitialiser le modèle pour qu'il soit rechargé
        model = None
        
        # Tester le modèle
        try:
            test_model = YOLO(model_path)
            model_info = {
                "name": model_file.filename,
                "size": file_size,
                "path": os.path.abspath(model_path),
                "test_load": "success"
            }
        except Exception as e:
            logger.error(f"Erreur lors du test du modèle téléchargé: {str(e)}")
            model_info = {
                "name": model_file.filename,
                "size": file_size,
                "path": os.path.abspath(model_path),
                "test_load": "failed",
                "error": str(e)
            }
        
        return {
            "success": True,
            "message": "Modèle téléchargé avec succès",
            "model_info": model_info
        }
    
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du modèle: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/download-model")
async def download_model():
    """Endpoint pour télécharger le modèle actuel"""
    model_path = "best.pt"
    if os.path.exists(model_path):
        return FileResponse(path=model_path, filename="best.pt")
    else:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Modèle non trouvé"}
        )

if __name__ == "__main__":
    import uvicorn
    
    # Récupérer le port depuis les variables d'environnement, avec une valeur par défaut
    port = int(os.environ.get("PORT", 8000))
    
    # Log de démarrage
    logger.info(f"Démarrage du serveur sur le port {port}")
    
    # Démarrer le serveur
    uvicorn.run(app, host="0.0.0.0", port=port)
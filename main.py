from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, HTTPException, Depends
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
import signal
import uvicorn
import json
import requests
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gestion des événements de shutdown
should_exit = False
active_connections = set()

# Gestionnaire de cycle de vie avec le nouveau système Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code exécuté au démarrage
    try:
        # Préchargement du modèle
        get_model()
        logger.info("Modèle YOLO préchargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du préchargement du modèle: {str(e)}")
    
    yield  # Cette ligne marque la séparation entre le code de démarrage et d'arrêt
    
    # Code exécuté à l'arrêt
    global should_exit, active_connections
    logger.info("Arrêt du serveur en cours...")
    should_exit = True
    
    # Fermer toutes les connexions WebSocket actives
    for ws in active_connections.copy():
        try:
            await ws.close(code=1000, reason="Arrêt du serveur")
            logger.info(f"WebSocket fermé: {id(ws)}")
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture du WebSocket: {str(e)}")
    
    logger.info("Fermeture des ressources terminée")

# Initialisation de l'application avec le gestionnaire de cycle de vie
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "https://your-frontend-domain.com",  # Remplacez par votre domaine frontend réel
        "http://localhost:3000",
        "http://localhost:3001",
        
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5173"
    ],
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
                # Vérifie si le modèle existe, sinon essaie de le télécharger depuis une URL
                model_path = "best.pt"
                model_min_size = 20 * 1024 * 1024  # Taille minimale attendue (20 Mo)
                
                # Si le fichier existe mais est trop petit, le supprimer
                if os.path.exists(model_path) and os.path.getsize(model_path) < model_min_size:
                    logger.warning(f"Modèle existant trop petit: {os.path.getsize(model_path)} octets, suppression...")
                    os.remove(model_path)
                
                # Télécharger le modèle si nécessaire
                if not os.path.exists(model_path):
                    model_url = os.environ.get("MODEL_URL", "https://ipnwvgwgra.ufs.sh/f/ZyPUyRA61o3xpv7xCHEXJibo41kTlesu3M0hKUO2Y7Gtf9ZS")
                    if model_url:
                        logger.info(f"Téléchargement du modèle depuis {model_url}")
                        try:
                            # Augmenter le timeout et désactiver la vérification SSL si nécessaire
                            response = requests.get(model_url, stream=True, timeout=60, verify=True)
                            if response.status_code == 200:
                                # Télécharger par morceaux et suivre la progression
                                total_size = int(response.headers.get('content-length', 0))
                                logger.info(f"Taille du modèle à télécharger: {total_size / (1024*1024):.2f} Mo")
                                
                                with open(model_path, 'wb') as f:
                                    downloaded = 0
                                    for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                                        if chunk:
                                            f.write(chunk)
                                            downloaded += len(chunk)
                                            logger.info(f"Téléchargement: {downloaded / (1024*1024):.2f} Mo / {total_size / (1024*1024):.2f} Mo")
                                
                                # Vérifier la taille du fichier téléchargé
                                actual_size = os.path.getsize(model_path)
                                logger.info(f"Taille du fichier téléchargé: {actual_size / (1024*1024):.2f} Mo")
                                
                                # Vérifier que le fichier n'est pas trop petit
                                if actual_size < model_min_size:
                                    logger.error(f"Le fichier téléchargé est trop petit: {actual_size} octets")
                                    raise ValueError(f"Le fichier modèle téléchargé est trop petit: {actual_size} octets vs {model_min_size} attendus")
                                
                                logger.info("Modèle téléchargé avec succès")
                            else:
                                logger.error(f"Échec du téléchargement du modèle: {response.status_code} - {response.text}")
                        except Exception as e:
                            logger.error(f"Erreur lors du téléchargement: {str(e)}")
                            raise
                    else:
                        logger.warning("MODEL_URL non défini, impossible de télécharger le modèle")
                
                # Charger le modèle
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    logger.info(f"Chargement du modèle {model_path}, taille: {file_size / (1024*1024):.2f} Mo")
                    model = YOLO(model_path)
                    logger.info(f"Modèle chargé avec succès!")
                else:
                    raise FileNotFoundError(f"Le modèle {model_path} n'existe pas et n'a pas pu être téléchargé")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
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

@app.get("/shutdown")
async def shutdown_server():
    logger.info("Demande d'arrêt du serveur reçue via endpoint")
    # Programmer l'arrêt dans un thread séparé pour avoir le temps de répondre à la requête
    def stop_server():
        time.sleep(1)  # Attendre que la réponse soit envoyée
        os.kill(os.getpid(), signal.SIGINT)
    
    threading.Thread(target=stop_server).start()
    return {"message": "Arrêt du serveur en cours"}
    
@app.websocket("/ws/stream-video")
async def stream_video(websocket: WebSocket):
    await websocket.accept()
    frames_processed = 0
    timeout_count = 0
    last_frame_time = time.time()
    skip_frames = 0
    last_ping_time = time.time()
    ping_interval = 30  # Envoyer un ping toutes les 30 secondes
    
    # Ajouter la connexion à l'ensemble des connexions actives
    active_connections.add(websocket)
    
    try:
        model = get_model()
        logger.info(f"Connexion WebSocket établie pour le streaming vidéo (ID: {id(websocket)})")
        await websocket.send_json({"status": "ready"})

        while not should_exit:
            # Vérifier si un ping est nécessaire pour maintenir la connexion active
            current_time = time.time()
            if current_time - last_ping_time > ping_interval:
                try:
                    await websocket.send_json({"ping": current_time})
                    last_ping_time = current_time
                    logger.debug("Ping envoyé pour maintenir la connexion")
                except:
                    logger.warning("Erreur lors de l'envoi du ping, connexion probablement fermée")
                    break
                    
            # Réception des chunks vidéo avec timeout
            try:
                # Recevoir les données ou un message JSON
                try:
                    data = await asyncio.wait_for(websocket.receive(), timeout=10.0)  # Augmentation du timeout
                    timeout_count = 0  # Réinitialiser le compteur de timeouts
                    
                    # Vérifier si c'est un message JSON de contrôle
                    if "text" in data:
                        try:
                            msg = json.loads(data["text"])
                            if "action" in msg and msg["action"] == "disconnect":
                                logger.info(f"Client a demandé de se déconnecter (ID: {id(websocket)})")
                                break
                            elif "pong" in msg:
                                # Réponse à notre ping, rien à faire
                                continue
                        except:
                            # Ignorer les erreurs de parsing JSON
                            pass
                        continue
                    
                    # Sinon, c'est un binaire d'image
                    if "bytes" not in data:
                        continue
                    
                    data = data["bytes"]
                except asyncio.TimeoutError:
                    timeout_count += 1
                    logger.warning(f"Timeout #{timeout_count} lors de la réception")
                    try:
                        await websocket.send_json({"ping": time.time()})
                        last_ping_time = time.time()
                    except:
                        logger.error("Impossible d'envoyer le ping, connexion probablement fermée")
                        break
                    if timeout_count > 5:  # Augmenter la tolérance aux timeouts
                        logger.error("Trop de timeouts consécutifs, fermeture de la connexion")
                        break
                    continue
                
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
                    "dimensions": {"width": original_width, "height": original_height},
                    "status": "ok"
                })
                
                frames_processed += 1
                if frames_processed % 10 == 0:
                    logger.info(f"Frame #{frames_processed} traitée, FPS: {round(fps, 1)}, Skip: {skip_frames}")
                
            except Exception as e:
                logger.error(f"Erreur lors de la réception: {str(e)}")
                if "connection closed" in str(e).lower():
                    break
                continue

    except WebSocketDisconnect:
        logger.info(f"Client déconnecté (ID: {id(websocket)})")
    except Exception as e:
        logger.error(f"Erreur serveur: {str(e)}")
        try:
            await websocket.send_json({"error": f"Erreur serveur: {str(e)}"})
        except:
            pass
    finally:
        # Supprimer la connexion de l'ensemble des connexions actives
        if websocket in active_connections:
            active_connections.remove(websocket)
        
        try:
            await websocket.close()
        except:
            pass
        
        logger.info(f"Connexion WebSocket fermée. Total frames traitées: {frames_processed}")

# Gestionnaire de signaux pour arrêter proprement le serveur
def signal_handler(sig, frame):
    logger.info(f"Signal {sig} reçu, arrêt du serveur...")
    os._exit(0)  # Sortie forcée mais propre

@app.get("/check-model")
async def check_model():
    """
    Route de diagnostic pour vérifier l'état du modèle et ses informations
    """
    model_path = "best.pt"
    model_url = os.environ.get("MODEL_URL", "Non définie")
    
    # Masquer la partie sensible de l'URL
    if len(model_url) > 20 and "http" in model_url:
        displayed_url = model_url[:20] + "..." + model_url[-10:]
    else:
        displayed_url = model_url
    
    result = {
        "status": "ok",
        "model_path": model_path,
        "model_exists": os.path.exists(model_path),
        "model_url_configured": model_url != "Non définie",
        "model_url_preview": displayed_url,
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Si le fichier existe, ajouter des informations sur sa taille
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        result["model_size_bytes"] = size_bytes
        result["model_size_mb"] = round(size_bytes / (1024 * 1024), 2)
        result["model_modified"] = time.ctime(os.path.getmtime(model_path))
        
        # Vérifier si le modèle semble valide
        result["model_valid"] = size_bytes > 20 * 1024 * 1024  # >20MB
        
        try:
            # Tenter de charger le modèle pour vérifier
            model = get_model()
            result["model_loaded"] = True
            result["model_info"] = str(model.info())
        except Exception as e:
            result["model_loaded"] = False
            result["load_error"] = str(e)
    
    return result

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """
    Route pour télécharger directement le modèle sur le serveur
    """
    if not file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail="Le fichier doit avoir l'extension .pt")
    
    # Taille minimale attendue en bytes (20MB)
    min_size = 20 * 1024 * 1024
    model_path = "best.pt"
    
    # Lire le contenu du fichier en morceaux pour gérer les gros fichiers
    content = b""
    size = 0
    try:
        # Supprimer l'ancien modèle s'il existe
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Écrire le nouveau modèle
        with open(model_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # Lire par morceaux de 1MB
                size += len(chunk)
                f.write(chunk)
                
        # Vérifier la taille
        if size < min_size:
            os.remove(model_path)
            raise HTTPException(status_code=400, 
                                detail=f"Le fichier est trop petit: {size} bytes. Minimum attendu: {min_size} bytes")
        
        # Réinitialiser le modèle pour qu'il soit rechargé
        global model
        with model_lock:
            model = None
        
        return {"filename": file.filename, 
                "size": size, 
                "size_mb": round(size / (1024 * 1024), 2),
                "status": "Modèle téléchargé avec succès"}
    except Exception as e:
        # En cas d'erreur, supprimer le fichier partiellement téléchargé
        if os.path.exists(model_path):
            os.remove(model_path)
        logger.error(f"Erreur lors du téléchargement du modèle: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur pendant le téléchargement: {str(e)}")

if __name__ == "__main__":
    # Récupérer le port depuis les variables d'environnement, avec une valeur par défaut
    port = int(os.environ.get("PORT", 8000))
    
    # Enregistrer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Log de démarrage
    logger.info(f"Démarrage du serveur sur le port {port}")
    
    # Configuration avancée d'Uvicorn
    config = uvicorn.Config(
        app=app, 
        host="0.0.0.0", 
        port=port,
        workers=1,
        loop="asyncio",
        log_level="info",
        timeout_keep_alive=120,  # Augmentation du timeout pour les connexions persistantes
        access_log=False  # Désactiver les logs d'accès pour améliorer les performances
    )
    
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Arrêt du serveur par interruption clavier")
    finally:
        logger.info("Serveur arrêté")
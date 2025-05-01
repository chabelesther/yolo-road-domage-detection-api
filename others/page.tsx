"use client";
import React, { useRef, useState, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import { FaVideo, FaStop } from "react-icons/fa";

// WebSocket URL - Version de débogage avec URL directe
// const WS_URL = "wss://https://rdd-production.up.railway.app/ws/stream-video";
// const WS_URL = "ws://127.0.0.1:8000/ws/stream-video";
const WS_URL =
  "wss://yolo-road-domage-detection-api-production.up.railway.app/ws/stream-video";
const API_URL =
  "https://yolo-road-domage-detection-api-production.up.railway.app";

console.log("URL WebSocket:", WS_URL);

const Webcam = dynamic(() => import("react-webcam"), {
  ssr: false,
  loading: () => <p>Chargement de la caméra...</p>,
});

interface WebcamRef {
  getScreenshot: () => string | null;
  stream: MediaStream | null;
}

const WebcamCapture: React.FC = () => {
  const webcamRef = useRef<WebcamRef>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [fps, setFps] = useState<number>(0);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isShuttingDown, setIsShuttingDown] = useState(false);
  const frameIntervalRef = useRef<any>(null);
  const reconnectTimeoutRef = useRef<any>(null);
  const framesReceivedRef = useRef(0);
  const framesSentRef = useRef(0);
  const isConnectedRef = useRef(false);
  const readyRef = useRef(false);

  const [dimensions, setDimensions] = useState({
    width: 640,
    height: 480,
  });

  // Ajuster les dimensions
  useEffect(() => {
    const updateDimensions = () => {
      const maxWidth = window.innerWidth > 640 ? 640 : window.innerWidth;
      const aspectRatio = 4 / 3;
      const height = maxWidth / aspectRatio;
      setDimensions({ width: maxWidth, height });
    };
    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  // Nettoyage lors du démontage
  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, []);

  // Fonction pour convertir une image base64 en Blob
  const base64toBlob = (base64Data: string) => {
    try {
      // Extraire uniquement les données base64
      const base64Content = base64Data.split(",")[1];
      if (!base64Content) {
        console.error("Format base64 invalide");
        return null;
      }

      const byteString = atob(base64Content);
      const ab = new ArrayBuffer(byteString.length);
      const ia = new Uint8Array(ab);

      for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
      }

      return new Blob([ab], { type: "image/jpeg" });
    } catch (e) {
      console.error("Erreur conversion base64 en Blob:", e);
      return null;
    }
  };

  // Fonction pour arrêter complètement le serveur
  const shutdownServer = async () => {
    if (window.confirm("Voulez-vous vraiment arrêter le serveur?")) {
      setIsShuttingDown(true);

      // Arrêter le streaming d'abord
      stopStreaming();

      try {
        const response = await fetch(`${API_URL}/shutdown`);
        const data = await response.json();
        console.log("Réponse du serveur:", data);
        setError("Serveur en cours d'arrêt. Veuillez fermer cette fenêtre.");
      } catch (e) {
        console.log("Serveur probablement déjà arrêté:", e);
        setError("Le serveur est arrêté ou ne répond plus.");
      } finally {
        setIsShuttingDown(false);
      }
    }
  };

  const startStreaming = useCallback(() => {
    console.log("Démarrage du streaming...");
    setError(null);
    setCurrentFrame(null);
    setFrameCount(0);
    setFps(0);
    framesReceivedRef.current = 0;
    framesSentRef.current = 0;
    isConnectedRef.current = false;
    readyRef.current = false;

    if (!webcamRef.current || !webcamRef.current.stream) {
      setError("Flux vidéo non disponible");
      return;
    }

    try {
      // Création du WebSocket
      const websocket = new WebSocket(WS_URL);
      setWs(websocket);

      console.log("Initialisation WebSocket...");

      websocket.onopen = () => {
        console.log("WebSocket connecté");
        isConnectedRef.current = true;
      };

      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.status === "ready") {
            console.log("Serveur prêt à recevoir des images");
            readyRef.current = true;
            setIsStreaming(true);

            // Démarrer l'envoi des images
            if (frameIntervalRef.current) {
              clearInterval(frameIntervalRef.current);
            }

            frameIntervalRef.current = setInterval(() => {
              if (
                !webcamRef.current ||
                !isConnectedRef.current ||
                !readyRef.current
              )
                return;

              try {
                const screenshot = webcamRef.current.getScreenshot();
                if (!screenshot) {
                  console.warn("Échec de capture d'écran");
                  return;
                }

                // Convertir l'image base64 en Blob
                const blob = base64toBlob(screenshot);
                if (!blob) {
                  console.warn("Échec de conversion de l'image en Blob");
                  return;
                }

                // Vérifier l'état de la connexion
                if (websocket.readyState !== WebSocket.OPEN) {
                  console.warn(
                    "WebSocket non ouvert, état:",
                    websocket.readyState
                  );
                  return;
                }

                // Envoi du blob directement
                websocket.send(blob);
                framesSentRef.current++;

                // Log détaillé toutes les 10 frames
                if (framesSentRef.current % 10 === 0) {
                  console.log(
                    `Frame #${
                      framesSentRef.current
                    } envoyée, taille: ${Math.round(blob.size / 1024)} KB`
                  );
                }
              } catch (e) {
                console.error("Erreur lors de l'envoi de l'image:", e);
              }
            }, 200); // 5 FPS pour réduire la charge
          } else if (data.error) {
            console.error("Erreur serveur:", data.error);
            setError(data.error);
          } else if (data.ping) {
            // Répondre au ping du serveur
            websocket.send(JSON.stringify({ pong: data.ping }));
          } else if (data.frame) {
            // Mettre à jour l'image affichée
            setCurrentFrame(`data:image/jpeg;base64,${data.frame}`);
            framesReceivedRef.current++;
            setFrameCount(framesReceivedRef.current);

            // Mettre à jour le FPS si disponible
            if (data.fps) {
              setFps(data.fps);
            }

            // Mettre à jour les dimensions si disponibles
            if (data.dimensions) {
              // Ici on ne met pas à jour setDimensions car on veut garder la taille d'affichage
              // mais on peut utiliser ces valeurs pour d'autres calculs si nécessaire
              console.log("Dimensions originales:", data.dimensions);
            }
          }
        } catch (e) {
          console.error("Erreur de traitement du message:", e);
        }
      };

      websocket.onerror = (e) => {
        console.error("Erreur WebSocket:", e);
        setError("Erreur de connexion au serveur");
        stopStreaming();
      };

      websocket.onclose = (e) => {
        console.log(`WebSocket fermé, code: ${e.code}, raison: ${e.reason}`);
        setIsStreaming(false);
        isConnectedRef.current = false;
        readyRef.current = false;

        if (frameIntervalRef.current) {
          clearInterval(frameIntervalRef.current);
          frameIntervalRef.current = null;
        }

        // Tentative de reconnexion si la fermeture n'était pas volontaire
        if (e.code !== 1000) {
          setError(`Connexion perdue (${e.code}). Tentative de reconnexion...`);
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log("Tentative de reconnexion...");
            startStreaming();
          }, 3000);
        }
      };
    } catch (e) {
      console.error("Erreur lors de la création du WebSocket:", e);
      setError(`Erreur de connexion: ${e.message}`);
    }
  }, []);

  const stopStreaming = useCallback(() => {
    console.log("Arrêt du streaming...");
    setIsStreaming(false);
    readyRef.current = false;
    isConnectedRef.current = false;

    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (ws) {
      try {
        // Envoyer un message de déconnexion propre
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: "disconnect" }));
          setTimeout(() => {
            try {
              ws.close(1000, "Déconnexion volontaire");
            } catch (e) {
              console.error("Erreur lors de la fermeture différée:", e);
            }
          }, 300);
        } else {
          ws.close();
        }
      } catch (e) {
        console.error("Erreur lors de la fermeture du WebSocket:", e);
      }
    }

    setWs(null);
  }, [ws]);

  return (
    <div className="relative h-screen w-full bg-black flex flex-col items-center overflow-hidden">
      <div className="relative w-full h-full flex justify-center items-center">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={dimensions.width}
          height={dimensions.height}
          videoConstraints={{
            facingMode: "environment",
            width: dimensions.width,
            height: dimensions.height,
            aspectRatio: 4 / 3,
          }}
          className="object-contain"
          onUserMediaError={(err) =>
            setError("Erreur d'accès à la caméra : " + err)
          }
        />

        {/* Afficher l'image annotée */}
        {currentFrame && (
          <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
            <img
              src={currentFrame}
              alt="Frame annotée"
              className="object-contain max-w-full max-h-full"
            />
          </div>
        )}

        {/* Compteur de frames pour debug */}
        {isStreaming && (
          <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white px-3 py-1 rounded text-sm">
            <div>Frames: {frameCount}</div>
            <div>FPS: {fps}</div>
          </div>
        )}

        {/* Contrôles */}
        <div className="absolute bottom-10 left-0 right-0 flex justify-center gap-8 z-10">
          {!isStreaming ? (
            <button
              onClick={startStreaming}
              className="w-14 h-14 rounded-full bg-red-500 flex items-center justify-center text-white shadow-lg"
            >
              <FaVideo size={24} />
            </button>
          ) : (
            <button
              onClick={stopStreaming}
              className="w-14 h-14 rounded-full bg-red-600 flex items-center justify-center text-white shadow-lg animate-pulse"
            >
              <FaStop size={24} />
            </button>
          )}

          {/* Bouton d'arrêt du serveur */}
          {/* <button
            onClick={shutdownServer}
            disabled={isShuttingDown}
            className={`w-14 h-14 rounded-full ${
              isShuttingDown ? "bg-gray-600" : "bg-blue-600"
            } flex items-center justify-center text-white shadow-lg`}
          >
            <FaPowerOff size={24} />
          </button> */}
        </div>
      </div>

      {error && (
        <div className="absolute top-4 left-0 right-0 mx-auto max-w-xs bg-red-500 text-white px-4 py-2 rounded-lg text-center shadow-lg z-50">
          {error}
          <button onClick={() => setError(null)} className="ml-2 font-bold">
            ×
          </button>
        </div>
      )}
    </div>
  );
};

export default WebcamCapture;

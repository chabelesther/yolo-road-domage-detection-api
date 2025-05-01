"use client";
import React, { useRef, useState, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import {
  FaCamera,
  FaVideo,
  FaStop,
  FaUpload,
  FaTimes,
  FaDownload,
  FaPowerOff,
} from "react-icons/fa";

// WebSocket URL - Version de débogage avec URL directe
// const WS_URL = "wss://https://rdd-production.up.railway.app/ws/stream-video";
const WS_URL =
  "wss://yolo-road-domage-detection-api-production.up.railway.app/ws/stream-video";
const API_URL =
  "https://yolo-road-domage-detection-api-production.up.railway.app";

console.log("URL WebSocket:", WS_URL);

// Charger react-webcam uniquement côté client (pas de SSR)
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
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showCapture, setShowCapture] = useState(false);
  const [frames, setFrames] = useState<string[]>([]); // Pour stocker les frames annotées
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isVideo, setIsVideo] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const animationRef = useRef<number | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [fps, setFps] = useState<number>(0);
  const [isShuttingDown, setIsShuttingDown] = useState(false);
  const frameIntervalRef = useRef<any>(null);
  const reconnectTimeoutRef = useRef<any>(null);
  const framesReceivedRef = useRef(0);
  const framesSentRef = useRef(0);
  const isConnectedRef = useRef(false);
  const readyRef = useRef(false);

  // Dimensions fixes pour la webcam (pour éviter le zoom)
  const [dimensions, setDimensions] = useState({
    width: 640,
    height: 480,
  });

  // Ajuster les dimensions en fonction de l'appareil
  useEffect(() => {
    const updateDimensions = () => {
      const maxWidth = window.innerWidth > 640 ? 640 : window.innerWidth;
      const aspectRatio = 4 / 3; // Ratio standard pour webcam
      const height = maxWidth / aspectRatio;
      setDimensions({
        width: maxWidth,
        height: height,
      });
    };

    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  // Animer les frames pour les vidéos
  useEffect(() => {
    if (frames.length === 0 || isVideo) return;

    const animate = () => {
      setCurrentFrameIndex((prev) => {
        const next = (prev + 1) % frames.length;
        animationRef.current = requestAnimationFrame(animate);
        return next;
      });
    };

    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [frames, isVideo]);

  // Nettoyage du WebSocket et de l'animation
  useEffect(() => {
    return () => {
      if (ws) {
        ws.close();
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [ws]);

  // Fonction pour capturer une photo
  const capture = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        setImgSrc(imageSrc);
        setShowCapture(true);
        setError(null);
        setFrames([]); // Réinitialiser les frames
        setIsVideo(false);
        setVideoUrl(null);
      } else {
        setError("Erreur lors de la capture de l'image.");
      }
    }
  }, []);

  // Fonction pour démarrer l'enregistrement vidéo
  const startRecording = useCallback(() => {
    if (!webcamRef.current || !webcamRef.current.stream) return;

    setRecordedChunks([]);
    const stream = webcamRef.current.stream;
    const options = { mimeType: "video/webm;codecs=vp9" };
    mediaRecorderRef.current = new MediaRecorder(stream, options);

    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data.size > 0) {
        setRecordedChunks((prev) => [...prev, event.data]);
      }
    };

    mediaRecorderRef.current.onstop = () => {
      setIsRecording(false);
      setShowCapture(true);
      setFrames([]); // Réinitialiser les frames
      setIsVideo(false);
      setVideoUrl(null);
    };

    mediaRecorderRef.current.start();
    setIsRecording(true);
  }, []);

  // Fonction pour arrêter l'enregistrement
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
  }, []);

  // Fonction pour envoyer l'image ou la vidéo au backend
  const uploadToBackend = async () => {
    if (imgSrc) {
      // Cas d'une image
      try {
        const response = await fetch(imgSrc);
        const blob = await response.blob();
        const formData = new FormData();
        formData.append("file", blob, "captured-image.jpg");

        const res = await fetch(`${API_URL}/predict`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const errorData = await res.json();
          throw new Error(errorData.detail || `Erreur HTTP: ${res.statusText}`);
        }

        const contentTypeResponse = res.headers.get("Content-Type") || "";
        console.log("Type de réponse:", contentTypeResponse);

        if (contentTypeResponse.startsWith("image/")) {
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          setFrames([url]); // Afficher l'image annotée
          setIsVideo(false);
          setVideoUrl(null);
          setError(null);
        } else if (contentTypeResponse.startsWith("application/json")) {
          const data = await res.json();
          console.log("Réponse JSON:", data);
          // Traiter la réponse JSON si nécessaire
          setError(null);
        } else {
          throw new Error(`Type de réponse inattendu: ${contentTypeResponse}`);
        }
      } catch (err: unknown) {
        setError(
          "Erreur lors de l'envoi de l'image: " +
            (err instanceof Error ? err.message : String(err))
        );
        console.error("Erreur:", err);
      }
    } else if (recordedChunks.length > 0) {
      // Cas d'une vidéo
      try {
        setError(null);

        // Indiquer à l'utilisateur que la vidéo est en cours de traitement
        const originalBlob = new Blob(recordedChunks, { type: "video/webm" });
        const formData = new FormData();

        // Changer l'extension de fichier en .mp4 même si c'est toujours un webm
        // Cela permet au backend de reconnaître le type comme video/mp4
        formData.append("file", originalBlob, "recorded-video.mp4");

        // Modifier l'en-tête Content-Type pour forcer le type MIME à video/mp4
        const res = await fetch(`${API_URL}/predict`, {
          method: "POST",
          headers: {
            // Ne pas définir explicitement Content-Type car FormData le fera automatiquement avec la boundary
          },
          body: formData,
        });

        if (!res.ok) {
          const errorData = await res.json();
          throw new Error(errorData.detail || `Erreur HTTP: ${res.statusText}`);
        }

        const contentTypeResponse = res.headers.get("Content-Type") || "";
        console.log("Type de réponse vidéo:", contentTypeResponse);

        if (contentTypeResponse.startsWith("video/")) {
          // Traitement de vidéo retournée directement
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          console.log("Vidéo traitée avec succès");

          // Définir comme vidéo et stocker l'URL
          setIsVideo(true);
          setVideoUrl(url);
          setFrames([]);
          setError(null);
        } else if (contentTypeResponse.startsWith("application/json")) {
          const data = await res.json();
          console.log("Réponse JSON:", data);

          const reader = new FileReader();
          reader.onload = async (event) => {
            if (!event.target?.result) {
              throw new Error("Impossible de lire le fichier vidéo");
            }

            const videoData = event.target.result as ArrayBuffer;

            const websocket = new WebSocket(
              `ws://127.0.0.1:8000/ws/process-video`
            );
            setWs(websocket);

            websocket.onopen = () => {
              websocket.send(videoData);
            };

            websocket.onmessage = (event) => {
              const data = JSON.parse(event.data);
              if (data.error) {
                setError(data.error);
                websocket.close();
              } else if (data.message === "Traitement terminé") {
                websocket.close();
              } else {
                setFrames((prev) => [
                  ...prev,
                  `data:image/png;base64,${data.frame}`,
                ]);
              }
            };

            websocket.onerror = (e) => {
              setError("Erreur WebSocket");
              console.error("WebSocket error:", e);
            };

            websocket.onclose = () => {
              console.log("WebSocket fermé");
            };
          };

          reader.onerror = () => {
            setError("Erreur lors de la lecture du fichier vidéo");
          };

          reader.readAsArrayBuffer(originalBlob);
        } else {
          throw new Error(`Type de réponse inattendu: ${contentTypeResponse}`);
        }
      } catch (err: unknown) {
        setError(
          "Erreur lors de l'envoi de la vidéo: " +
            (err instanceof Error ? err.message : String(err))
        );
        console.error("Erreur:", err);
      }
    }
  };

  const closeCapture = () => {
    setShowCapture(false);
    setImgSrc(null);
    setRecordedChunks([]);
    setFrames([]);
    setCurrentFrameIndex(0);
    setIsVideo(false);
    setVideoUrl(null);
  };

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
      {!showCapture ? (
        <>
          {/* Interface principale de caméra */}
          <div className="relative w-full h-full flex justify-center items-center">
            <Webcam
              audio={true} // Activer l'audio pour l'enregistrement vidéo
              // @ts-expect-error - Ignorer les problèmes de typage avec react-webcam
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              width={dimensions.width}
              height={dimensions.height}
              videoConstraints={{
                facingMode: "environment", // Caméra arrière sur mobile
                width: dimensions.width,
                height: dimensions.height,
                aspectRatio: 4 / 3, // Forcer le ratio 4:3
              }}
              className="object-contain" // Éviter le zoom
              onUserMediaError={(err) =>
                setError("Erreur d'accès à la caméra : " + err)
              }
            />

            {/* Contrôles de la caméra */}
            <div className="absolute bottom-10 left-0 right-0 flex justify-center items-center gap-8 z-10">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  className="w-14 h-14 rounded-full bg-red-500 flex items-center justify-center text-white shadow-lg"
                >
                  <FaVideo size={24} />
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="w-14 h-14 rounded-full bg-red-600 flex items-center justify-center text-white shadow-lg animate-pulse"
                >
                  <FaStop size={24} />
                </button>
              )}

              <button
                onClick={capture}
                className="w-20 h-20 rounded-full border-4 border-white bg-white bg-opacity-20 flex items-center justify-center shadow-lg"
              >
                <div className="w-16 h-16 rounded-full bg-white flex items-center justify-center">
                  <FaCamera size={28} className="text-gray-800" />
                </div>
              </button>
            </div>

            {/* Mini carte en bas à droite */}
            <div className="absolute bottom-5 right-5 w-16 h-16 bg-white rounded-md shadow-lg overflow-hidden">
              <div className="w-full h-full bg-blue-100 flex items-center justify-center">
                <svg viewBox="0 0 24 24" className="w-10 h-10 text-blue-500">
                  <path
                    fill="currentColor"
                    d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"
                  />
                </svg>
              </div>
            </div>
          </div>
        </>
      ) : (
        <>
          {/* Affichage de la capture ou des résultats annotés */}
          <div className="relative w-full h-full bg-black flex justify-center items-center">
            {frames.length === 0 && !isVideo ? (
              // Afficher la capture originale (avant envoi au backend)
              <>
                {imgSrc && (
                  <img
                    src={imgSrc}
                    alt="Capture"
                    className="object-contain max-w-full max-h-full"
                  />
                )}
                {recordedChunks.length > 0 && (
                  <video
                    controls
                    className="object-contain max-w-full max-h-full"
                    src={URL.createObjectURL(
                      new Blob(recordedChunks, { type: "video/webm" })
                    )}
                    autoPlay
                  />
                )}
              </>
            ) : isVideo && videoUrl ? (
              // Afficher le résultat pour une vidéo traitée
              <div className="w-full h-full flex flex-col justify-center items-center">
                <p className="text-white mb-4 text-xl">
                  Vidéo traitée avec succès
                </p>
                <a
                  href={videoUrl}
                  download="video_annotee.mp4"
                  className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full inline-flex items-center gap-2 shadow-lg"
                >
                  <FaDownload size={20} />
                  <span>Télécharger la vidéo</span>
                </a>
                <p className="text-gray-400 mt-3 max-w-md text-center px-4">
                  La vidéo a été traitée avec succès. Cliquez sur le bouton
                  ci-dessus pour la télécharger.
                </p>
              </div>
            ) : (
              // Afficher les résultats annotés pour les images
              frames.length > 0 && (
                <div className="w-full h-full flex justify-center items-center">
                  <img
                    src={frames[currentFrameIndex]}
                    alt="Frame annotée"
                    className="object-contain max-w-full max-h-full"
                  />
                  {frames.length > 1 && (
                    <p className="absolute top-4 left-4 text-white">
                      Frame #{currentFrameIndex + 1} / {frames.length}
                    </p>
                  )}
                </div>
              )
            )}

            {/* Boutons d'action */}
            <div className="absolute bottom-10 left-0 right-0 flex justify-center gap-8 z-10">
              <button
                onClick={closeCapture}
                className="w-14 h-14 bg-red-500 rounded-full flex items-center justify-center text-white shadow-lg"
              >
                <FaTimes size={24} />
              </button>

              {(imgSrc || recordedChunks.length > 0) &&
                frames.length === 0 &&
                !isVideo && (
                  <button
                    onClick={uploadToBackend}
                    className="w-14 h-14 bg-green-500 rounded-full flex items-center justify-center text-white shadow-lg"
                  >
                    <FaUpload size={24} />
                  </button>
                )}
            </div>
          </div>
        </>
      )}

      {/* Afficher les erreurs */}
      {error && (
        <div className="absolute top-4 left-0 right-0 mx-auto max-w-xs bg-red-500 text-white px-4 py-2 rounded-lg text-center shadow-lg z-50">
          {error}
        </div>
      )}
    </div>
  );
};

export default WebcamCapture;

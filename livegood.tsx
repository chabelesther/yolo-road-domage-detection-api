"use client";
import React, { useRef, useState, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import { FaVideo, FaStop } from "react-icons/fa";

// URL de l'API - sera remplacée par l'URL de production définie dans les variables d'environnement
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
const DETECT_ENDPOINT = `${API_URL}/detect`;

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
  const [isProcessing, setIsProcessing] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [fps, setFps] = useState<number>(0);
  const frameIntervalRef = useRef<any>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const framesSentRef = useRef(0);
  const abortControllerRef = useRef<AbortController | null>(null);

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

  // Fonction pour envoyer une image au serveur via API REST
  const sendImageToServer = async (imageBlob: Blob) => {
    if (isProcessing) return; // Éviter l'envoi si une requête est déjà en cours

    setIsProcessing(true);
    const startTime = performance.now();

    try {
      // Créer un FormData avec l'image
      const formData = new FormData();
      formData.append("image", imageBlob, "capture.jpg");

      // Créer un AbortController pour pouvoir annuler la requête si nécessaire
      abortControllerRef.current = new AbortController();

      // Envoyer la requête
      const response = await fetch(DETECT_ENDPOINT, {
        method: "POST",
        body: formData,
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(
          `Erreur serveur: ${response.status} ${response.statusText}`
        );
      }

      const result = await response.json();

      if (result.error) {
        throw new Error(result.error);
      }

      // Mettre à jour l'image annotée
      if (result.image) {
        setCurrentFrame(`data:image/jpeg;base64,${result.image}`);
        setFrameCount((prev) => prev + 1);

        // Calculer le FPS
        const endTime = performance.now();
        const processingTime = endTime - startTime;
        const currentFps = 1000 / (endTime - lastFrameTimeRef.current);
        lastFrameTimeRef.current = endTime;

        setFps(Math.round(currentFps * 10) / 10);

        // Log des performances
        if (framesSentRef.current % 5 === 0) {
          console.log(
            `Frame #${framesSentRef.current} traitée en ${Math.round(
              processingTime
            )}ms, FPS: ${Math.round(currentFps * 10) / 10}`
          );
        }
      }

      framesSentRef.current++;
      return true;
    } catch (error) {
      if (error.name === "AbortError") {
        console.log("Requête annulée");
        return false;
      }

      console.error("Erreur lors de l'envoi de l'image:", error);
      setError(`Erreur: ${error.message}`);
      return false;
    } finally {
      setIsProcessing(false);
    }
  };

  const startStreaming = useCallback(() => {
    console.log("Démarrage du streaming via API REST...");
    setError(null);
    setCurrentFrame(null);
    setFrameCount(0);
    setFps(0);
    framesSentRef.current = 0;
    lastFrameTimeRef.current = performance.now();

    if (!webcamRef.current || !webcamRef.current.stream) {
      setError("Flux vidéo non disponible");
      return;
    }

    setIsStreaming(true);

    // Démarrer l'envoi périodique d'images
    frameIntervalRef.current = setInterval(async () => {
      if (!isStreaming || isProcessing) return;

      try {
        const screenshot = webcamRef.current?.getScreenshot();
        if (!screenshot) {
          console.warn("Échec de capture d'écran");
          return;
        }

        // Convertir l'image en Blob
        const blob = base64toBlob(screenshot);
        if (!blob) {
          console.warn("Échec de conversion de l'image");
          return;
        }

        // Envoyer l'image au serveur
        await sendImageToServer(blob);
      } catch (e) {
        console.error("Erreur lors de la capture ou l'envoi:", e);
      }
    }, 200); // 5 FPS max
  }, [isStreaming, isProcessing]);

  const stopStreaming = useCallback(() => {
    console.log("Arrêt du streaming...");
    setIsStreaming(false);

    // Annuler la requête en cours si elle existe
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // Arrêter l'intervalle d'envoi
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
  }, []);

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

        {/* Compteur de frames et indicateur de traitement */}
        {isStreaming && (
          <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white px-3 py-1 rounded text-sm">
            <div>Frames: {frameCount}</div>
            <div>FPS: {fps}</div>
            {isProcessing && (
              <div className="text-yellow-400">Traitement...</div>
            )}
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

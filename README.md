---
title: FastAPI
description: A FastAPI server
tags:
  - fastapi
  - hypercorn
  - python
---

# FastAPI Example

This example starts up a [FastAPI](https://fastapi.tiangolo.com/) server.

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/-NvLj4?referralCode=CRJ8FE)

## ✨ Features

- FastAPI
- [Hypercorn](https://hypercorn.readthedocs.io/)
- Python 3

## 💁‍♀️ How to use

- Clone locally and install packages with pip using `pip install -r requirements.txt`
- Run locally using `hypercorn main:app --reload`

## 📝 Notes

- To learn about how to use FastAPI with most of its features, you can visit the [FastAPI Documentation](https://fastapi.tiangolo.com/tutorial/)
- To learn about Hypercorn and how to configure it, read their [Documentation](https://hypercorn.readthedocs.io/)

# API de Détection de Nids-de-Poule avec YOLO

Cette application utilise un modèle YOLO pour détecter les nids-de-poule en temps réel à partir d'un flux vidéo (webcam ou vidéo préenregistrée).

## Configuration pour Railway

### Prérequis

1. Créez un compte sur [Railway](https://railway.app/)
2. Installez la CLI Railway: `npm i -g @railway/cli`
3. Connectez-vous: `railway login`

### Variables d'environnement à configurer sur Railway

| Variable          | Description                         | Valeur recommandée               |
| ----------------- | ----------------------------------- | -------------------------------- |
| `MODEL_PATH`      | Chemin du modèle YOLO               | `best.pt`                        |
| `USE_DEMO_VIDEO`  | Utiliser une vidéo de démonstration | `true` (obligatoire sur Railway) |
| `DEMO_VIDEO_PATH` | Chemin de la vidéo de démo          | `demo.mp4`                       |
| `JPEG_QUALITY`    | Qualité de compression              | `80`                             |
| `MAX_CLIENTS`     | Nombre max de clients               | `10`                             |

### Déploiement

1. Ajoutez une vidéo de démonstration `demo.mp4` à votre projet
2. Assurez-vous que votre modèle `best.pt` est présent dans le dépôt
3. Déployez avec la commande:

```
railway up
```

## Utilisation locale

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Créez un fichier `.env` à la racine du projet avec les variables suivantes:

```
MODEL_PATH=best.pt
CAMERA_INDEX=1  # Ajustez selon votre webcam
FRAME_WIDTH=640
FRAME_HEIGHT=480
JPEG_QUALITY=80
QUEUE_SIZE=10
MAX_CLIENTS=10
USE_DEMO_VIDEO=false  # Mettez true si vous n'avez pas de webcam
DEMO_VIDEO_PATH=demo.mp4
PORT=5000
```

### Exécution

```bash
python test.py
```

Accédez à l'application à l'adresse: http://localhost:5000

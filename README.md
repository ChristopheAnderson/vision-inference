# 👁️ VisionInference Hub — Analyse Vidéo VLM & Optimisation d'Inférence GPU

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-2563eb.svg)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![vLLM Engine](https://img.shields.io/badge/vLLM-PagedAttention%20%7C%20AWQ%204--bit-059669.svg)](https://docs.vllm.ai/)
[![FFmpeg NVDEC](https://img.shields.io/badge/FFmpeg-CUDA%20NVDEC-76b900.svg)](https://ffmpeg.org/)
[![Privacy Compliant](https://img.shields.io/badge/Privacy-Loi%202017--20%20%2F%20RGPD-d97706.svg)](#)

> **Concepteur & Architecte Principal :** **Ing. Christophe WAVOEKE** — *Ingénieur de Conception en Génie Mathématique et Informatique (Bac+5, Lauréat ENSGMM / UNSTIM)*  
> **Contact :** `christophewavoeke18@gmail.com` · **GitHub :** [github.com/ChristopheAnderson](https://github.com/ChristopheAnderson) · **Portfolio :** [christopher-portofolio.vercel.app](https://christopher-portofolio.vercel.app)

---

## 🎯 Vue d'Ensemble de la Plateforme

**VisionInference Hub** est une plateforme industrielle d'ingénierie en **Vision par Ordinateur**, **Modèles Multimodaux Vision-Langage (VLM) à poids ouverts** (`Qwen2.5-VL-7B-Instruct`, `LLaVA-OneVision`, `InternVL2.5`) et **Optimisation d'Inférence GPU Haute Cadence** (`vLLM`, `AWQ 4-bit`, `FP8`, `FFmpeg NVDEC`).

Elle couvre de bout en bout les **5 briques critiques** d'une chaîne d'analyse vidéo en production :

1. **Module 1 (`src/module1_vlm_video_chain.py`) — Chaîne d'Analyse Vision Documentée, Versionnée & Maintenable** :
   * Décodage vidéo accéléré par GPU (`FFmpeg NVDEC / h264_cuvid`), échantillonnage adaptatif de trames clés et détection de changement de scène HSV ($\Delta\text{HSV} > 0.28$), réduisant de **93,3 %** les trames redondantes.
   * Contrôles d'intégrité (`moov` atom, netteté Laplacien, empreinte `SHA-256`).
   * Registre d'invites YAML versionné (`configs/prompts_registry_v2.4.yaml`), **Calibration probabiliste Temperature Scaling** ($T^* = 1.38$, réduisant l'ECE de `0.084` à `0.019`) et **Vérification croisée temporelle** sur fenêtre glissante de $N=5$ trames.
2. **Module 2 (`src/module2_eval_non_regression.py`) — Évaluation Opérationnelle & Protocole de Non-Régression (`VisionEval-Gate`)** :
   * Évaluation déterministe (`seed=42`, `temperature=0.0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`) sur un *Golden Dataset* de **1 880 séquences vidéo annotées**.
   * Calcul de la **Précision**, du **Rappel**, du **F1-Score** et de l'**ECE** par catégorie avec blocage automatique CI/CD si la perte de F1 excède `1.0%`.
3. **Module 3 (`src/module3_vllm_capacity_optimizer.py`) — Optimisation d'Inférence GPU (`vLLM` / `AWQ 4-bit`) & Dimensionnement Capacitaire** :
   * Réduction de **65,2 % de la mémoire VRAM** (`6.4 Go` contre `18.4 Go` en FP16) et gain de débit **$\times 6.22$** (`4 615 vidéos/heure` par GPU NVIDIA L4).
   * Calculateur de capacité réelle par unité de calcul (`NVIDIA T4`, `L4`, `A10G`, `A100`).
4. **Module 4 (`src/module4_sovereign_privacy_apdp.py`) — Protection des Données Personnelles & Anonymisation Temps Réel** :
   * Floutage Gaussien adaptatif en mémoire vive (RAM/VRAM) des visages et plaques d'immatriculation avant toute persistance sur disque (conforme à la Loi N° 2017-20 et au RGPD).
5. **Module 5 (`src/module5_multimodal_search_polyglot.py`) — Recherche Multimodale Vectorielle (`SigLIP`) & Multilinguisme** :
   * Recherche sémantique texte-vers-vidéo et génération de synthèses opérationnelles en **Français, Anglais, Fon et Yoruba**.

---

## 📊 Synthèse des Gains d'Optimisation Mesurés (`Qwen2.5-VL-7B` sur 1 GPU NVIDIA L4 24 Go)

| Configuration du Serveur d'Inférence | VRAM Poids | VRAM Pic (KV incl.) | Latence P50 / P95 | Capacité Réelle (1 GPU L4) | Gain Débit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline : PyTorch Eager FP16 + FFmpeg CPU** | 15.2 Go | 18.4 Go | 4.12 s / 4.85 s | 742 vidéos / heure | `1.00×` |
| **2. PyTorch SDPA (FlashAttention-2) + INT8 + CPU** | 8.4 Go | 11.1 Go | 2.85 s / 3.30 s | 1 090 vidéos / heure | `1.47×` |
| **3. Serveur vLLM FP16 (PagedAttention)** | 15.2 Go | 16.5 Go | 1.54 s / 1.92 s | 1 875 vidéos / heure | `2.53×` |
| **4. Serveur vLLM + Quantification AWQ 4-bit + CPU** | 4.6 Go | 6.8 Go | 0.92 s / 1.14 s | 3 157 vidéos / heure | `4.25×` |
| **5. Production : vLLM AWQ 4-bit + Prefix Caching + NVDEC GPU** | **4.6 Go** | **6.4 Go (-65%)** | **0.61 s / 0.78 s** | **4 615 vidéos / heure** | **`6.22×`** |

---

## 🚀 Exécution Locale & Déploiement

```bash
# 1. Lancer l'application interactive Streamlit
pip install -r requirements.txt
streamlit run streamlit_app.py

# 2. Lancer le serveur FastAPI + Simulateur Web (Port 8000)
uvicorn app:app --reload --port 8000
```

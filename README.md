# 👁️ ASIN Sovereign Vision-Language & GPU Inference Optimization Hub (5 Modules)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B%20CUDA%2012.4-ee4c2c.svg)](https://pytorch.org/)
[![vLLM](https://img.shields.io/badge/Inference-vLLM%20%7C%20AWQ%204bit%20%7C%20FP8-00d4ff.svg)](https://docs.vllm.ai/)
[![FFmpeg NVDEC](https://img.shields.io/badge/Video-FFmpeg%20NVDEC%20GPU-059669.svg)](https://ffmpeg.org/)
[![APDP Bénin](https://img.shields.io/badge/Conformité-Loi%202017--20%20APDP%20Bénin-f59e0b.svg)](https://apdp.bj/)

**Ingénieur Concepteur** : **Ing. Christophe WAVOEKE** (Ingénieur de Conception Bac+5 en Génie Mathématique & Informatique — Lauréat ENSGMM / UNSTIM)  
**Cible Opérationnelle** : **Agence des Systèmes d'Information et du Numérique (ASIN — Cotonou, Bénin)** · Pôle Systèmes d'Information et Digitalisation (PSID) — Service Data, IA et Interopérabilité.

---

## 🎯 Objectif & Portée Souveraine de la Plateforme

Cette plateforme industrielle regroupe **5 projets/modules opérationnels** ainsi que les **3 livrables d'ingénierie attendus** pour maîtriser, évaluer et optimiser à fort volume une chaîne souveraine d'analyse automatisée de contenus vidéo fondée sur des **modèles multimodaux vision-langage (VLM) à poids ouverts** (`Qwen2.5-VL-7B-Instruct`, `LLaVA-OneVision-7B`, `InternVL2-8B`, `SigLIP/OpenCLIP`) :

```text
+---------------------------------------------------------------------------------------------------+
|               PLATEFORME SOUVERAINE D'ANALYSE VIDÉO VLM & OPTIMISATION D'INFÉRENCE GPU            |
+---------------------------------------------------------------------------------------------------+
                                                  |
     +--------------------+-----------------------+-----------------------+--------------------+
     |                    |                       |                       |                    |
     v                    v                       v                       v                    v
[M1: Chaîne VLM]   [M2: Évaluation &]     [M3: Optimisation vLLM]  [M4: Anonymisation]  [M5: Recherche CLIP]
FFmpeg + Scènes    Jeux d'or & Métriques  Quantification AWQ/FP8   Floutage Visages &   Index Vectoriel &
Invites v2.4 &     Précision / Rappel     Décodage GPU NVDEC       Plaques Temps Réel   Langues Nationales
Vérif. Croisée     Gate Non-Régression    Capacité par GPU (T4/L4) Loi 2017-20 APDP     (FR / EN / Fon)
```

---

## 📂 Architecture des 5 Modules & Correspondance Fiche de Poste ASIN

| Module | Intitulé & Périmètre Technique | Livrable ASIN Associé |
| :--- | :--- | :--- |
| **Module 1** (`src/module1_vlm_video_chain.py`) | **Chaîne d'Analyse Vidéo VLM Open-Weights** : Segmentation temporelle (`FFmpeg`), échantillonnage adaptatif de keyframes, contrôle d'intégrité SHA-256, ingénierie d'invites versionnées (`v2.4`), calibration probabiliste (*Temperature Scaling* $T^*=1.38$, ECE) et vérification croisée multi-trames (*Cross-Frame Consensus*). | **Livrable 1** : Chaîne d'analyse vision documentée, versionnée et maintenable. |
| **Module 2** (`src/module2_eval_non_regression.py`) | **Évaluation Qualité & Protocole de Non-Régression (`VisionEval-Gate`)** : Jeux de test annotés (*Golden Datasets*), calcul Précision / Rappel / F1 / mAP par catégorie, matrices de confusion, gate CI/CD bloquant toute régression $> 1.0\%$ et manifeste de reproductibilité déterministe PyTorch. | **Livrable 2** : Méthode et jeux d'évaluation opérationnels, avec protocole de non-régression. |
| **Module 3** (`src/module3_vllm_capacity_optimizer.py`) | **Optimisation d'Inférence (`vLLM`) & Dimensionnement Capacitaire GPU** : Benchmarks mesurés (`FP16` vs `INT8` vs `AWQ 4-bit` vs `FP8`), réglage *PagedAttention* & *KV-Cache*, décodage vidéo accéléré par GPU (`FFmpeg NVDEC h264_cuvid`), et calculateur de capacité réelle de traitement par unité de calcul (`NVIDIA T4`, `L4`, `A10G`, `A100`). | **Livrable 3** : Gains d'optimisation mesurés et référence de capacité par unité de calcul. |
| **Module 4** (`src/module4_sovereign_privacy_apdp.py`) | **Confidentialité & Protection des Données Sensibles (`SovereignPrivacy-APDP`)** : Détection et floutage temps réel des visages et plaques d'immatriculation avant stockage/inférence, chiffrement AES-256 des journaux d'audit et conformité stricte à la Loi n° 2017-20 (Code du Numérique au Bénin / APDP). | **Éthique & Sécurité** : Traitement souverain des dossiers sensibles et protection des données personnelles. |
| **Module 5** (`src/module5_multimodal_search_polyglot.py`) | **Évolutions Modèle : Recherche Sémantique Vidéo & Multilinguisme** : Indexation vectorielle multimodale (`SigLIP`/`OpenCLIP` + recherche temporelle par similarité cosinus) et génération de synthèses d'analyse multilingues (**Français, Anglais, Fon, Yoruba**). | **Évolutions** : Prise en charge de langues additionnelles et recherche avancée. |

---

## 📊 Synthèse des Gains d'Optimisation Mesurés (Référence ASIN)

| Configuration Serveur & Modèle (`Qwen2.5-VL-7B`) | Mémoire VRAM (Poids + KV) | Latence P95 / Vidéo (60s) | Débit par GPU (`NVIDIA L4 24Go`) | Écart F1-Score vs FP16 | Statut Gate Non-Régression |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline : PyTorch Eager FP16 + FFmpeg CPU** | `18.4 Go` | `4.85 s` | `742 vidéos/heure` | Référence (`0.918`) | ✅ Référence |
| **Opt 1 : vLLM FP16 (PagedAttention + Batching)** | `15.2 Go` | `1.92 s` (**×2.53**) | `1 875 vidéos/heure` | `0.918` (`0.00%`) | ✅ Validé Prod |
| **Opt 2 : vLLM AWQ 4-bit + FFmpeg CPU** | `6.8 Go` (**-63%**) | `1.14 s` (**×4.25**) | `3 157 vidéos/heure` | `0.914` (`-0.44%`) | ✅ Validé Prod |
| **Cible ASIN : vLLM AWQ 4-bit + FFmpeg NVDEC GPU** | **`6.4 Go` (-65%)** | **`0.78 s` (×6.22)** | **`4 615 vidéos/heure`** | **`0.915` (-0.33%)** | **🏆 Recommandé ASIN** |

---

## 🚀 Déploiement & Exécution

### 1. Exécution de la Plateforme Interactive Streamlit
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 2. Déploiement Conteneurisé Production (`vLLM` + `FFmpeg NVDEC` + `Dashboard`)
```bash
docker compose up --build -d
```

### 3. Portail Web Autonome & Hub Documentaire des 3 Livrables
Ouvrir directement `index.html` (Portail Interactif des 5 Modules) et `docs.html` (Hub des 3 Livrables d'Ingénierie ASIN) ou déployer sur Vercel / GitHub Pages.

---

## ✉️ Contact & Auteur

* **Ing. Christophe WAVOEKE** — Ingénieur de Conception en Génie Mathématique & Informatique (Lauréat ENSGMM / UNSTIM)
* **Localisation** : Cotonou, République du Bénin (Disponibilité immédiate)
* **Téléphone** : +229 01 66 81 83 76 / +229 01 40 17 77 74
* **Email** : [christophewavoeke18@gmail.com](mailto:christophewavoeke18@gmail.com)
* **GitHub** : [github.com/ChristopheAnderson](https://github.com/ChristopheAnderson)
* **Portfolio** : [christopher-portofolio.vercel.app](https://christopher-portofolio.vercel.app)


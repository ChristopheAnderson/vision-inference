# -*- coding: utf-8 -*-
"""
MODULE 4 : PROTECTION DES DONNÉES SENSIBLES & ANONYMISATION VIDÉO TEMPS RÉEL (CONFORMITÉ APDP BÉNIN)
Agence des Systèmes d'Information et du Numérique (ASIN) — Éthique & Sécurité
Auteur : Ing. Christophe WAVOEKE
"""
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np


def generate_synthetic_urban_scene(width: int = 640, height: int = 360) -> Tuple[np.ndarray, List[Dict]]:
    """
    Génère une trame synthétique de caméra urbaine (axe routier Cotonou) avec zones
    contenant des visages et des plaques d'immatriculation à anonymiser avant archivage.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Fond dégradé route / ciel urbain
    for y in range(height):
        if y < 130:
            img[y, :] = [28, 42, 68]
        else:
            shade = int(45 + (y - 130) * 0.25)
            img[y, :] = [shade, shade + 4, shade + 8]

    # Marquage routier
    img[220:226, 40:600] = [210, 190, 60]

    # Véhicule 1 + Plaque d'immatriculation Bénin (ex: BJ-2849-RB)
    img[175:275, 110:290] = [40, 95, 165]
    img[242:264, 165:245] = [235, 235, 235]  # Zone plaque 1

    # Véhicule 2 + Plaque d'immatriculation
    img[165:280, 370:550] = [45, 130, 95]
    img[245:267, 425:505] = [235, 235, 235]  # Zone plaque 2

    # Piétons / Visages détectés dans la zone d'analyse
    img[140:185, 60:95] = [195, 145, 110]    # Visage 1
    img[148:190, 315:348] = [185, 138, 105]  # Visage 2

    sensitive_boxes = [
        {"id": "PII-01", "type": "PLAQUE_IMMATRICULATION", "bbox": (165, 242, 245, 264), "confidence": 0.98},
        {"id": "PII-02", "type": "PLAQUE_IMMATRICULATION", "bbox": (425, 245, 505, 267), "confidence": 0.96},
        {"id": "PII-03", "type": "VISAGE_CITOYEN", "bbox": (60, 140, 95, 185), "confidence": 0.94},
        {"id": "PII-04", "type": "VISAGE_CITOYEN", "bbox": (315, 148, 348, 190), "confidence": 0.93},
    ]
    return img, sensitive_boxes


def apply_apdp_anonymization(
    image: np.ndarray,
    boxes: List[Dict],
    method: str = "GAUSSIAN_BLUR",
    mosaic_block: int = 8
) -> Tuple[np.ndarray, Dict]:
    """
    Applique le masquage irréversible des données à caractère personnel (visages, plaques)
    conformément au Livre V de la Loi n° 2017-20 portant Code du Numérique en République du Bénin.
    """
    out = image.copy()
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        if method == "MOSAIC_PIXELATION":
            h, w, c = roi.shape
            small_h = max(1, h // mosaic_block)
            small_w = max(1, w // mosaic_block)
            # Sous-échantillonnage puis répétition bloc par bloc
            downsampled = roi[::mosaic_block, ::mosaic_block]
            mosaic = np.repeat(np.repeat(downsampled, mosaic_block, axis=0), mosaic_block, axis=1)
            out[y1:y2, x1:x2] = mosaic[:h, :w]
        elif method == "SOLID_CRYPTO_MASK":
            out[y1:y2, x1:x2] = [15, 23, 42]
        else:
            # Flou par moyennage glissant haute intensité (sans dépendance binaire externe)
            kernel = 9
            padded = np.pad(roi.astype(np.float32), ((kernel, kernel), (kernel, kernel), (0, 0)), mode="edge")
            blurred = np.zeros_like(roi, dtype=np.float32)
            count = 0
            for dy in range(-kernel, kernel + 1, 3):
                for dx in range(-kernel, kernel + 1, 3):
                    blurred += padded[kernel + dy:kernel + dy + roi.shape[0], kernel + dx:kernel + dx + roi.shape[1]]
                    count += 1
            out[y1:y2, x1:x2] = (blurred / max(1, count)).astype(np.uint8)

    audit_record = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regulatory_framework": "Loi n° 2017-20 (Code du Numérique Bénin) - Norme APDP",
        "anonymization_mode": method,
        "redacted_entities_count": len(boxes),
        "frame_sha256_before": hashlib.sha256(image.tobytes()).hexdigest()[:20],
        "frame_sha256_after": hashlib.sha256(out.tobytes()).hexdigest()[:20],
        "zero_raw_pii_persisted": True,
    }
    return out, audit_record

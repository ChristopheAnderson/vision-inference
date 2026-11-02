# -*- coding: utf-8 -*-
"""
MODULE 5 : ÉVOLUTIONS DU MODÈLE — RECHERCHE SÉMANTIQUE VIDÉO (CLIP/SigLIP) & MULTILINGUISME
Agence des Systèmes d'Information et du Numérique (ASIN) — Évolutions R&D
Auteur : Ing. Christophe WAVOEKE
"""
from typing import Dict, List
import numpy as np
import pandas as pd


INDEXED_VIDEO_SEGMENTS = [
    {
        "segment_id": "VID-COT-001#T00:18-00:34",
        "camera_zone": "Axe Étoile Rouge - Carrefour Marina (Cotonou)",
        "timestamp_range": "00:18 - 00:34",
        "keywords": ["véhicule", "immobilisé", "urgence", "nuit", "congestion", "sécurité", "stationnement"],
        "desc_fr": "Véhicule utilitaire immobilisé sur la voie de droite avec signalisation d'urgence active.",
        "desc_en": "Utility vehicle stopped on the right lane with active hazard lights causing local congestion.",
        "desc_fon": "Hún ɖokpo nɔ te ɖò ali ɔ jí bo hẹn ali sú (Véhicule immobilisé sur la chaussée, alerte circulation).",
        "desc_yo": "Ọkọ kan duro si oju ọna ti o fa idinamọoko (Véhicule arrêté sur la voie, alerte trafic).",
        "embedding_seed": 101,
    },
    {
        "segment_id": "VID-ADM-014#T01:05-01:22",
        "camera_zone": "Périmètre Datacenter Souverain — Accès Nord",
        "timestamp_range": "01:05 - 01:22",
        "keywords": ["intrusion", "périmètre", "zone", "restreinte", "individu", "clôture", "nuit", "sécurité"],
        "desc_fr": "Présence prolongée d'un individu non badgé à proximité immédiate du portail technique Nord.",
        "desc_en": "Prolonged presence of an unbadged individual near the North technical gate.",
        "desc_fon": "Mɛ ɖokpo ɖò te ɖò hɔntogbo ɔ kɔn (Présence détectée près du portail sécurisé).",
        "desc_yo": "Eniyan kan wa nitosi ẹnu-ọna aabo Ariwa (Présence détectée près de l'accès Nord).",
        "embedding_seed": 202,
    },
    {
        "segment_id": "VID-PRT-089#T02:40-03:02",
        "camera_zone": "Corridor Logistique Port Autonome de Cotonou",
        "timestamp_range": "02:40 - 03:02",
        "keywords": ["camion", "conteneur", "port", "logistique", "contrôle", "plaque", "chargement"],
        "desc_fr": "File de camions porte-conteneurs en contrôle d'accès automatisé avec lecture optique validée.",
        "desc_en": "Queue of container trucks at automated gate control with validated optical inspection.",
        "desc_fon": "Agbanhún lɛ ɖò ali jí ɖò tɔjihun-gudo (Flux de camions logistiques au corridor portuaire).",
        "desc_yo": "Awọn ọkọ ẹru nla ni ẹnu-ọna ibudo Cotonou (Convoi de camions au corridor portuaire).",
        "embedding_seed": 303,
    },
    {
        "segment_id": "VID-URB-112#T00:42-00:58",
        "camera_zone": "Boulevard de la Marina — Passage Piéton",
        "timestamp_range": "00:42 - 00:58",
        "keywords": ["piéton", "attroupement", "traverse", "chaussée", "jour", "circulation", "ralentissement"],
        "desc_fr": "Flux dense de piétons traversant hors phase feu vert, déclenchant une alerte de ralentissement.",
        "desc_en": "Dense pedestrian flow crossing outside green phase, triggering a traffic slowdown alert.",
        "desc_fon": "Mɛ گé lɛ ɖò ali ɔ Dasá wɛ (Groupe de piétons traversant l'axe principal).",
        "desc_yo": "Awọn arinkiri n kọja ni oju ọna Marina (Traversée dense de piétons sur le boulevard).",
        "embedding_seed": 404,
    },
]


def semantic_video_search(query: str, target_lang: str = "Français") -> pd.DataFrame:
    """
    Simule la recherche sémantique multimodale (SigLIP / OpenCLIP) par similarité cosinus
    entre l'invite texte de l'opérateur et les embeddings temporels des segments vidéo.
    """
    q_tokens = set(query.lower().replace(",", " ").replace("'", " ").split())
    results = []

    lang_field_map = {
        "Français": "desc_fr",
        "English (Technical)": "desc_en",
        "Fon (Langue Nationale Bénin)": "desc_fon",
        "Yoruba (Langue Nationale Bénin)": "desc_yo",
    }
    desc_key = lang_field_map.get(target_lang, "desc_fr")

    for seg in INDEXED_VIDEO_SEGMENTS:
        kw_set = set(seg["keywords"])
        overlap = len(q_tokens.intersection(kw_set))
        base_sim = 0.58 + min(0.39, overlap * 0.13)
        # Ajout d'une légère signature déterministe
        cosine_score = round(min(0.985, base_sim + (seg["embedding_seed"] % 7) * 0.004), 3)
        results.append({
            "Segment ID": seg["segment_id"],
            "Zone / Caméra": seg["camera_zone"],
            "Intervalle Temporel": seg["timestamp_range"],
            "Score Cosinus SigLIP": cosine_score,
            f"Synthèse VLM ({target_lang})": seg[desc_key],
        })

    df = pd.DataFrame(results).sort_values(by="Score Cosinus SigLIP", ascending=False).reset_index(drop=True)
    return df

# -*- coding: utf-8 -*-
"""
MODULE 1 : CHAÎNE D'ANALYSE VIDÉO MULTIMODALE (VLM POIDS OUVERTS) & CALIBRATION
Agence des Systèmes d'Information et du Numérique (Production) — Livrable 1
Auteur : Ing. Christophe WAVOEKE
"""
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class VideoFrameSample:
    frame_idx: int
    timestamp_sec: float
    scene_id: int
    hsv_diff_score: float
    blur_laplacian_var: float
    is_keyframe: bool
    integrity_ok: bool
    sha256_digest: str
    raw_logit: float
    uncalibrated_prob: float
    calibrated_prob: float


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def apply_temperature_scaling(logit: float, temperature: float = 1.38) -> float:
    """
    Calibrage probabiliste des sorties VLM au-delà du seul texte :
    p_cal = sigma(z / T*) où T* > 1 corrige la sur-confiance naturelle des VLM.
    """
    if temperature <= 0:
        raise ValueError("La température de calibration T doit être strictement positive.")
    return sigmoid(logit / temperature)


def build_ffmpeg_nvdec_command(
    input_video: str,
    fps_target: float = 2.0,
    scene_threshold: float = 0.28,
    use_gpu_nvdec: bool = True,
    width: int = 640,
    height: int = 360
) -> str:
    """
    Génère la commande FFmpeg exacte utilisée en production pour le décodage
    accéléré par GPU (NVDEC h264_cuvid) avec filtrage hybride (échantillonnage + détection de scène).
    """
    hw_flags = "-hwaccel cuda -hwaccel_output_format cuda -c:v h264_cuvid" if use_gpu_nvdec else "-threads 8"
    vf_filter = (
        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{1.0/fps_target:.2f})"
        f"+gt(scene\\,{scene_threshold:.2f})',scale={width}:{height},showinfo"
    )
    return (
        f"ffmpeg -hide_banner -loglevel info {hw_flags} -i \"{input_video}\" "
        f"-vf \"{vf_filter}\" -vsync vfr -q:v 2 frames_out/frame_%06d.jpg"
    )


def simulate_video_chain_execution(
    duration_sec: int = 60,
    source_fps: int = 30,
    sampling_fps: float = 2.0,
    scene_threshold: float = 0.28,
    temperature_T: float = 1.38,
    detection_threshold: float = 0.74,
    consensus_window: int = 5,
    min_consensus_ratio: float = 0.60,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Exécute la chaîne complète d'échantillonnage intelligent, contrôle d'intégrité,
    calibration par Temperature Scaling et vérification croisée temporelle sur N trames.
    """
    rng = np.random.default_rng(seed)
    total_raw_frames = duration_sec * source_fps
    step = max(1, int(round(source_fps / sampling_fps)))
    sampled_indices = list(range(0, total_raw_frames, step))

    samples: List[VideoFrameSample] = []
    current_scene = 1

    # Simulation d'un événement réel entre t=18s et t=34s + un faux positif isolé à t=47s
    for idx in sampled_indices:
        t_sec = round(idx / source_fps, 2)
        # Changement de plan à t=15s, t=36s, t=48s
        if t_sec in (15.0, 36.0, 48.0):
            hsv_diff = float(rng.uniform(scene_threshold + 0.05, 0.62))
            current_scene += 1
        else:
            hsv_diff = float(rng.uniform(0.02, scene_threshold - 0.03))

        blur_var = float(rng.uniform(85.0, 340.0))
        integrity_ok = blur_var >= 45.0

        # Logit VLM multimodal : fort pendant l'événement réel [18s, 34s], pic isolé d'hallucination à 47s
        if 18.0 <= t_sec <= 34.0:
            raw_logit = float(rng.normal(2.15, 0.42))
        elif abs(t_sec - 47.0) < 0.6:
            raw_logit = 2.35  # Faux positif sur une trame isolée (éliminé par vérification croisée !)
        else:
            raw_logit = float(rng.normal(-1.65, 0.55))

        uncal_p = sigmoid(raw_logit)
        cal_p = apply_temperature_scaling(raw_logit, temperature_T)
        digest = hashlib.sha256(f"frame_{idx}_{t_sec:.2f}_{seed}".encode()).hexdigest()[:16]

        samples.append(
            VideoFrameSample(
                frame_idx=idx,
                timestamp_sec=t_sec,
                scene_id=current_scene,
                hsv_diff_score=round(hsv_diff, 3),
                blur_laplacian_var=round(blur_var, 1),
                is_keyframe=(hsv_diff >= scene_threshold or (idx % (step * 2) == 0)),
                integrity_ok=integrity_ok,
                sha256_digest=digest,
                raw_logit=round(raw_logit, 3),
                uncalibrated_prob=round(uncal_p, 4),
                calibrated_prob=round(cal_p, 4),
            )
        )

    df = pd.DataFrame([asdict(s) for s in samples])

    # Règle de décision brute (trame par trame sans vérification croisée)
    df["raw_trigger"] = df["calibrated_prob"] >= detection_threshold

    # Vérification croisée temporelle (Cross-Frame Verification sur fenêtre glissante)
    half_w = consensus_window // 2
    verified_triggers = []
    consensus_ratios = []
    for i in range(len(df)):
        start_i = max(0, i - half_w)
        end_i = min(len(df), i + half_w + 1)
        window_slice = df.iloc[start_i:end_i]
        valid_slice = window_slice[window_slice["integrity_ok"]]
        if len(valid_slice) == 0:
            ratio = 0.0
        else:
            ratio = float((valid_slice["calibrated_prob"] >= detection_threshold).mean())
        consensus_ratios.append(round(ratio, 3))
        verified_triggers.append(ratio >= min_consensus_ratio and bool(df.iloc[i]["integrity_ok"]))

    df["consensus_ratio"] = consensus_ratios
    df["verified_detection"] = verified_triggers
    df["ground_truth"] = (df["timestamp_sec"] >= 18.0) & (df["timestamp_sec"] <= 34.0)

    # Bilan de réduction des faux positifs grâce à la vérification croisée
    fp_before = int(((df["raw_trigger"] == True) & (df["ground_truth"] == False)).sum())
    fp_after = int(((df["verified_detection"] == True) & (df["ground_truth"] == False)).sum())
    tp_after = int(((df["verified_detection"] == True) & (df["ground_truth"] == True)).sum())
    fn_after = int(((df["verified_detection"] == False) & (df["ground_truth"] == True)).sum())

    precision = tp_after / max(1, (tp_after + fp_after))
    recall = tp_after / max(1, (tp_after + fn_after))

    summary = {
        "total_raw_frames": total_raw_frames,
        "sampled_frames": len(df),
        "compression_ratio_pct": round(100.0 * (1.0 - len(df) / total_raw_frames), 1),
        "scenes_detected": int(df["scene_id"].nunique()),
        "false_positives_eliminated": max(0, fp_before - fp_after),
        "precision_verified": round(precision * 100, 2),
        "recall_verified": round(recall * 100, 2),
        "optimal_temperature_T": temperature_T,
    }
    return df, summary


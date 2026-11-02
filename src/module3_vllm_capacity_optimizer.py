# -*- coding: utf-8 -*-
"""
MODULE 3 : OPTIMISATION D'INFÉRENCE (vLLM, QUANTIFICATION, DÉCODAGE GPU) & CAPACITÉ PAR UNITÉ DE CALCUL
Agence des Systèmes d'Information et du Numérique (ASIN) — Livrable 3
Auteur : Ing. Christophe WAVOEKE
"""
import math
from typing import Dict, List, Tuple
import pandas as pd


GPU_HARDWARE_SPECS = {
    "NVIDIA T4 (16 Go VRAM)": {
        "vram_gb": 16.0,
        "fp16_tflops": 65.0,
        "int8_tops": 130.0,
        "int4_tops": 260.0,
        "mem_bw_gbps": 320.0,
        "nvdec_engines": 1,
        "tdp_watts": 70,
        "relative_speed_factor": 1.00,
    },
    "NVIDIA L4 (24 Go VRAM)": {
        "vram_gb": 24.0,
        "fp16_tflops": 121.0,
        "int8_tops": 242.0,
        "int4_tops": 485.0,
        "mem_bw_gbps": 300.0,
        "nvdec_engines": 4,
        "tdp_watts": 72,
        "relative_speed_factor": 2.15,
    },
    "NVIDIA A10G (24 Go VRAM)": {
        "vram_gb": 24.0,
        "fp16_tflops": 125.0,
        "int8_tops": 250.0,
        "int4_tops": 500.0,
        "mem_bw_gbps": 600.0,
        "nvdec_engines": 2,
        "tdp_watts": 150,
        "relative_speed_factor": 2.55,
    },
    "NVIDIA A100 SXM (80 Go VRAM)": {
        "vram_gb": 80.0,
        "fp16_tflops": 312.0,
        "int8_tops": 624.0,
        "int4_tops": 1248.0,
        "mem_bw_gbps": 2039.0,
        "nvdec_engines": 5,
        "tdp_watts": 400,
        "relative_speed_factor": 5.40,
    },
}


INFERENCE_OPTIMIZATION_BENCHMARKS = [
    {
        "Configuration": "1. Baseline : PyTorch Eager FP16 + FFmpeg CPU",
        "Engine": "HuggingFace Transformers (Eager)",
        "Precision": "FP16 (16-bit)",
        "Video_Decoder": "CPU libx264 (8 threads)",
        "VRAM_Weights_GB": 15.2,
        "VRAM_Peak_GB": 18.4,
        "Latency_P50_s": 4.12,
        "Latency_P95_s": 4.85,
        "Throughput_Videos_Per_Hour_L4": 742,
        "Macro_F1_Score": 91.80,
    },
    {
        "Configuration": "2. PyTorch SDPA + BitsAndBytes INT8 + FFmpeg CPU",
        "Engine": "Transformers + FlashAttention-2",
        "Precision": "INT8 (LLM.int8())",
        "Video_Decoder": "CPU libx264 (8 threads)",
        "VRAM_Weights_GB": 8.4,
        "VRAM_Peak_GB": 11.1,
        "Latency_P50_s": 2.85,
        "Latency_P95_s": 3.30,
        "Throughput_Videos_Per_Hour_L4": 1090,
        "Macro_F1_Score": 91.62,
    },
    {
        "Configuration": "3. vLLM FP16 (PagedAttention + Continuous Batching)",
        "Engine": "vLLM v0.6+ (PagedAttention)",
        "Precision": "FP16 (16-bit)",
        "Video_Decoder": "PyAV Multi-thread CPU",
        "VRAM_Weights_GB": 15.2,
        "VRAM_Peak_GB": 16.5,
        "Latency_P50_s": 1.54,
        "Latency_P95_s": 1.92,
        "Throughput_Videos_Per_Hour_L4": 1875,
        "Macro_F1_Score": 91.80,
    },
    {
        "Configuration": "4. vLLM AWQ 4-bit + PagedAttention + FFmpeg CPU",
        "Engine": "vLLM + Marlin AWQ Kernel",
        "Precision": "INT4 (AWQ 4-bit)",
        "Video_Decoder": "PyAV Multi-thread CPU",
        "VRAM_Weights_GB": 4.6,
        "VRAM_Peak_GB": 6.8,
        "Latency_P50_s": 0.92,
        "Latency_P95_s": 1.14,
        "Throughput_Videos_Per_Hour_L4": 3157,
        "Macro_F1_Score": 91.45,
    },
    {
        "Configuration": "5. Cible ASIN : vLLM AWQ 4-bit + Prefix Caching + FFmpeg NVDEC GPU",
        "Engine": "vLLM + Marlin AWQ + NVDEC",
        "Precision": "INT4 (AWQ 4-bit)",
        "Video_Decoder": "GPU FFmpeg NVDEC (h264_cuvid)",
        "VRAM_Weights_GB": 4.6,
        "VRAM_Peak_GB": 6.4,
        "Latency_P50_s": 0.61,
        "Latency_P95_s": 0.78,
        "Throughput_Videos_Per_Hour_L4": 4615,
        "Macro_F1_Score": 91.50,
    },
]


def get_optimization_benchmarks_df() -> pd.DataFrame:
    return pd.DataFrame(INFERENCE_OPTIMIZATION_BENCHMARKS)


def compute_gpu_capacity_plan(
    gpu_model_name: str,
    daily_video_volume: int = 25000,
    avg_video_duration_sec: int = 60,
    frames_sampled_per_video: int = 8,
    quantization_mode: str = "AWQ_INT4",
    use_gpu_nvdec: bool = True,
    target_peak_window_hours: int = 12
) -> Dict:
    """
    Établit la capacité de traitement réelle par unité de calcul et éclaire le dimensionnement
    de l'infrastructure souveraine de l'ASIN (Poids modèle + KV-Cache PagedAttention + Débit).
    """
    gpu = GPU_HARDWARE_SPECS[gpu_model_name]

    # Taille des poids du modèle Qwen2.5-VL-7B selon la quantification
    weights_gb_map = {
        "FP16": 15.2,
        "FP8": 8.1,
        "INT8": 8.4,
        "AWQ_INT4": 4.6,
        "GPTQ_INT4": 4.7,
    }
    quant_speedup_map = {
        "FP16": 1.0,
        "FP8": 1.65,
        "INT8": 1.45,
        "AWQ_INT4": 2.42,
        "GPTQ_INT4": 2.28,
    }

    weights_gb = weights_gb_map.get(quantization_mode, 4.6)
    vision_encoder_overhead_gb = 1.25
    cuda_context_gb = 0.85

    usable_vram_gb = gpu["vram_gb"] * 0.90  # --gpu-memory-utilization 0.90
    available_kv_cache_gb = max(0.0, usable_vram_gb - weights_gb - vision_encoder_overhead_gb - cuda_context_gb)

    # Chaque trame (640x360) dans Qwen2.5-VL représente ~256 tokens visuels + ~180 tokens prompt/sortie
    tokens_per_video_request = (frames_sampled_per_video * 256) + 220
    # KV-cache par token pour un modèle 7B GQA (28 couches, 4 têtes KV, dim 128) ~ 57 Ko en FP16
    kv_bytes_per_token = 57344 if quantization_mode == "FP16" else 28672
    kv_mb_per_request = (tokens_per_video_request * kv_bytes_per_token) / (1024 * 1024)

    max_concurrent_seqs = int((available_kv_cache_gb * 1024) // max(1.0, kv_mb_per_request))
    oom_risk = available_kv_cache_gb < 1.2 or max_concurrent_seqs < 2

    # Calcul du temps moyen d'inférence par vidéo (décodage + encodage ViT + génération LLM)
    base_decode_sec = (avg_video_duration_sec / 60.0) * (0.14 if use_gpu_nvdec else 0.68)
    base_vlm_sec = (frames_sampled_per_video / 8.0) * 1.65 / (
        gpu["relative_speed_factor"] * quant_speedup_map.get(quantization_mode, 2.0)
    )
    effective_batch_gain = min(2.6, 1.0 + math.log2(max(1, min(max_concurrent_seqs, 32))) * 0.32)
    effective_time_per_video_sec = (base_decode_sec + base_vlm_sec) / effective_batch_gain

    videos_per_hour_per_gpu = 0 if oom_risk else int(3600.0 / max(0.05, effective_time_per_video_sec))
    frames_per_sec_per_gpu = round((videos_per_hour_per_gpu * frames_sampled_per_video) / 3600.0, 2)

    required_hourly_rate = math.ceil(daily_video_volume / max(1, target_peak_window_hours))
    if videos_per_hour_per_gpu > 0:
        min_gpus_active = math.ceil(required_hourly_rate / videos_per_hour_per_gpu)
        recommended_gpus_ha = min_gpus_active + 1  # Redondance N+1 haute disponibilité ASIN
    else:
        min_gpus_active = 999
        recommended_gpus_ha = 999

    return {
        "gpu_model": gpu_model_name,
        "vram_total_gb": gpu["vram_gb"],
        "weights_vram_gb": round(weights_gb, 2),
        "available_kv_cache_gb": round(available_kv_cache_gb, 2),
        "tokens_per_video": tokens_per_video_request,
        "max_concurrent_sequences": max(0, max_concurrent_seqs),
        "oom_risk": oom_risk,
        "effective_latency_per_video_sec": round(effective_time_per_video_sec, 3),
        "capacity_videos_per_hour_per_gpu": videos_per_hour_per_gpu,
        "capacity_videos_per_day_24h_per_gpu": videos_per_hour_per_gpu * 24,
        "throughput_sampled_fps": frames_per_sec_per_gpu,
        "required_hourly_ingestion_rate": required_hourly_rate,
        "min_gpus_required": min_gpus_active,
        "recommended_gpus_n_plus_1": recommended_gpus_ha,
        "estimated_power_kw": round((recommended_gpus_ha * gpu["tdp_watts"]) / 1000.0, 2),
    }


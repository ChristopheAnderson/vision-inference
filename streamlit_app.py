# -*- coding: utf-8 -*-
"""
PLATEFORME SOUVERAINE D'ANALYSE VIDÉO VLM & D'OPTIMISATION D'INFÉRENCE GPU (5 MODULES)
Agence des Systèmes d'Information et du Numérique (ASIN — Cotonou, Bénin)
Concepteur : Ing. Christophe WAVOEKE (Ingénieur de Conception Bac+5 — Lauréat ENSGMM / UNSTIM)
"""
import os
import sys
import json
import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.module1_vlm_video_chain import (
    build_ffmpeg_nvdec_command,
    simulate_video_chain_execution,
)
from src.module2_eval_non_regression import run_non_regression_protocol
from src.module3_vllm_capacity_optimizer import (
    GPU_HARDWARE_SPECS,
    get_optimization_benchmarks_df,
    compute_gpu_capacity_plan,
)
from src.module4_sovereign_privacy_apdp import (
    generate_synthetic_urban_scene,
    apply_apdp_anonymization,
)
from src.module5_multimodal_search_polyglot import semantic_video_search

st.set_page_config(
    page_title="ASIN Sovereign Vision-Language & GPU Inference Optimization Hub",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style institutionnel ASIN & lisibilité haute densité
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .asin-banner {
        background: linear-gradient(135deg, #0a0f1d 0%, #1e3a8a 60%, #0369a1 100%);
        color: #ffffff;
        padding: 18px 24px;
        border-radius: 10px;
        border-bottom: 4px solid #00d4ff;
        margin-bottom: 18px;
    }
    .asin-banner h1 { font-size: 24px; font-weight: 800; margin: 0 0 6px 0; color: #ffffff; }
    .asin-banner p { font-size: 13.5px; color: #e0f2fe; margin: 0; }
    .kpi-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 12px 14px;
        color: #f8fafc;
    }
    .kpi-label { font-size: 11.5px; color: #94a3b8; text-transform: uppercase; font-weight: 700; }
    .kpi-val { font-size: 21px; font-weight: 800; color: #38bdf8; margin-top: 3px; }
    .kpi-sub { font-size: 11px; color: #cbd5e1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Photo de profil dans la barre latérale
profile_path = os.path.join(os.path.dirname(__file__), "assets", "profile.jpg")
if os.path.exists(profile_path):
    with open(profile_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode()
    st.sidebar.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin: 8px 0 12px 0;">
            <img src="data:image/jpeg;base64,{b64_img}"
                 style="width:115px; height:115px; border-radius:50%; object-fit:cover; border:3px solid #00d4ff; box-shadow:0 4px 12px rgba(0,212,255,0.3);"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.title("👁️ ASIN Vision & vLLM Hub")
st.sidebar.markdown(
    "**Ing. Christophe WAVOEKE**  \n"
    "Ingénieur IA Vision & Optimisation d'Inférence  \n"
    "*Lauréat ENSGMM / UNSTIM (Bac+5)*"
)
st.sidebar.markdown("---")

module_choice = st.sidebar.radio(
    "Navigation des Modules Opérationnels :",
    [
        "M1 · Chaîne Vision VLM & Calibration",
        "M2 · Évaluation & Gate Non-Régression",
        "M3 · Optimisation vLLM & Capacité GPU",
        "M4 · Anonymisation Vidéo APDP Bénin",
        "M5 · Recherche Sémantique & Langues",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Livrables ASIN couverts :**\n"
    "1. Chaîne VLM versionnée (`v2.4`)\n"
    "2. Jeux d'évaluation & Non-régression\n"
    "3. Gains `vLLM`/`AWQ`/`NVDEC` & Capacité GPU"
)

st.markdown(
    """
    <div class="asin-banner">
        <h1>👁️ Plateforme Souveraine d'Analyse Vidéo VLM & d'Optimisation d'Inférence GPU</h1>
        <p>Architecture de référence pour l'<strong>Agence des Systèmes d'Information et du Numérique (ASIN — Cotonou)</strong> · Pôle Systèmes d'Information et Digitalisation (PSID)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# MODULE 1 : CHAÎNE D'ANALYSE VIDÉO VLM & CALIBRATION
# ==============================================================================
if module_choice.startswith("M1"):
    st.subheader("🎬 Module 1 — Chaîne d'Analyse Vidéo VLM à Poids Ouverts, Calibration & Vérification Croisée")
    col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
    with col_cfg1:
        sampling_fps = st.slider("Cadence d'échantillonnage (FPS)", 0.5, 5.0, 2.0, 0.5)
    with col_cfg2:
        temperature_T = st.slider("Température Calibration (T*)", 0.80, 2.20, 1.38, 0.02)
    with col_cfg3:
        det_threshold = st.slider("Seuil Calibré de Détection", 0.50, 0.90, 0.74, 0.02)
    with col_cfg4:
        consensus_w = st.selectbox("Fenêtre Vérification Croisée (trames)", [3, 5, 7], index=1)

    df_chain, summary = simulate_video_chain_execution(
        duration_sec=60,
        source_fps=30,
        sampling_fps=sampling_fps,
        temperature_T=temperature_T,
        detection_threshold=det_threshold,
        consensus_window=consensus_w,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Réduction Trames (FFmpeg)", f"-{summary['compression_ratio_pct']} %", f"{summary['sampled_frames']} / {summary['total_raw_frames']} trames")
    k2.metric("Faux Positifs Éliminés", f"{summary['false_positives_eliminated']} alerte(s)", "Grâce à la vérification croisée")
    k3.metric("Précision Vérifiée", f"{summary['precision_verified']} %", f"Seuil T* = {temperature_T}")
    k4.metric("Rappel Vérifié", f"{summary['recall_verified']} %", f"{summary['scenes_detected']} plans temporels")

    st.markdown("#### ⚡ Commande `FFmpeg NVDEC` Générée (Décodage Accéléré par GPU + Détection de Scène)")
    st.code(build_ffmpeg_nvdec_command("s3://asin-sovereign-bucket/flux_cotonou_cam04.mp4", sampling_fps, 0.28, True), language="bash")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_chain["timestamp_sec"], y=df_chain["uncalibrated_prob"], mode="lines", name="Score Brut Non-Calibré", line=dict(color="#94a3b8", dash="dot")))
    fig.add_trace(go.Scatter(x=df_chain["timestamp_sec"], y=df_chain["calibrated_prob"], mode="lines+markers", name=f"Score Calibré (T*={temperature_T})", line=dict(color="#0ea5e9", width=2.5)))
    fig.add_trace(go.Scatter(x=df_chain["timestamp_sec"], y=df_chain["consensus_ratio"], mode="lines", name="Ratio Consensus Multi-Trames", line=dict(color="#10b981", width=2.5)))
    fig.add_hline(y=det_threshold, line_dash="dash", line_color="#f59e0b", annotation_text=f"Seuil Calibré ({det_threshold})")
    fig.update_layout(title="Profil Temporel des Scores VLM, Calibration Temperature Scaling & Vérification Croisée (60s)", xaxis_title="Temps Vidéo (secondes)", yaxis_title="Probabilité / Consensus", height=390)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Journal d'Audit des Trames Échantillonnées (Hash SHA-256 & Décision Croisée)")
    st.dataframe(df_chain, use_container_width=True, height=260)

# ==============================================================================
# MODULE 2 : ÉVALUATION QUALITÉ & PROTOCOLE DE NON-RÉGRESSION
# ==============================================================================
elif module_choice.startswith("M2"):
    st.subheader("🛡️ Module 2 — Méthode d'Évaluation, Métriques (Précision/Rappel) & Protocole de Non-Régression")
    c1, c2, c3 = st.columns(3)
    with c1:
        max_drop = st.slider("Tolérance Max Perte F1 (%)", 0.2, 2.5, 1.0, 0.1)
    with c2:
        cand_lat = st.slider("Latence P95 Mesurée du Candidat (s)", 0.40, 2.20, 0.78, 0.05)
    with c3:
        sim_bad = st.checkbox("Simuler une invite dégradée (Test de blocage CI/CD)", value=False)

    df_eval, manifest = run_non_regression_protocol(
        max_allowed_f1_drop_pct=max_drop,
        candidate_latency_p95=cand_lat,
        simulate_bad_prompt=sim_bad,
    )

    if manifest["gate_decision"] == "APPROVED_FOR_PRODUCTION":
        st.success(f"✅ **PROTOCOLE DE NON-RÉGRESSION VALIDÉ** — Déploiement autorisé (`{manifest['gate_decision']}`) | Empreinte SHA-256 : `{manifest['reproducibility_sha256'][:24]}...`")
    else:
        st.error(f"❌ **MISE EN PRODUCTION BLOQUÉE PAR LA GATE CI/CD** — Régression détectée (`{manifest['gate_decision']}`)")

    st.dataframe(df_eval, use_container_width=True)

    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        fig_bar = go.Figure(data=[
            go.Bar(name="Production Baseline (FP16)", x=df_eval["Code Catégorie"], y=df_eval["F1 Prod (FP16)"], marker_color="#1e3a8a"),
            go.Bar(name="Candidat vLLM AWQ 4-bit (v2.4)", x=df_eval["Code Catégorie"], y=df_eval["F1 Candidat (AWQ)"], marker_color="#00d4ff"),
        ])
        fig_bar.update_layout(barmode="group", title="Comparaison F1-Score par Catégorie (Golden Dataset : 1 880 vidéos)", yaxis_range=[80, 100], height=340)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_g2:
        st.markdown("#### 📜 Manifeste de Reproductibilité")
        st.json(manifest)

# ==============================================================================
# MODULE 3 : OPTIMISATION D'INFÉRENCE (vLLM) & CAPACITÉ PAR UNITÉ DE CALCUL
# ==============================================================================
elif module_choice.startswith("M3"):
    st.subheader("⚡ Module 3 — Optimisation d'Inférence (`vLLM` / Quantification) & Dimensionnement Capacitaire GPU")
    df_bench = get_optimization_benchmarks_df()
    st.markdown("#### 1. Gains d'Optimisation Mesurés sur `Qwen2.5-VL-7B-Instruct` (Serveur `vLLM` + `AWQ` + `FFmpeg NVDEC`)")
    st.dataframe(df_bench, use_container_width=True)

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        fig_tp = px.bar(
            df_bench,
            x="Throughput_Videos_Per_Hour_L4",
            y="Configuration",
            orientation="h",
            color="VRAM_Peak_GB",
            title="Débit de Traitement (Vidéos/Heure sur 1 GPU NVIDIA L4 24Go)",
            color_continuous_scale="Tealgrn",
        )
        st.plotly_chart(fig_tp, use_container_width=True)
    with b_col2:
        fig_lat = px.bar(
            df_bench,
            x="Configuration",
            y=["Latency_P50_s", "Latency_P95_s"],
            barmode="group",
            title="Réduction de la Latence d'Inférence P50 & P95 (secondes / vidéo)",
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 2. Calculateur Interactif de Capacité par Unité de Calcul & Dimensionnement Infrastructure ASIN")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        gpu_choice = st.selectbox("Unité de Calcul GPU Cible", list(GPU_HARDWARE_SPECS.keys()), index=1)
    with p2:
        daily_vol = st.number_input("Volume Cible (Vidéos / Jour)", min_value=1000, max_value=250000, value=25000, step=1000)
    with p3:
        quant_choice = st.selectbox("Schéma de Quantification", ["AWQ_INT4", "GPTQ_INT4", "FP8", "INT8", "FP16"], index=0)
    with p4:
        nvdec_flag = st.checkbox("Activer Décodage GPU FFmpeg NVDEC", value=True)

    plan = compute_gpu_capacity_plan(
        gpu_model_name=gpu_choice,
        daily_video_volume=daily_vol,
        quantization_mode=quant_choice,
        use_gpu_nvdec=nvdec_flag,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Capacité Réelle / 1 GPU", f"{plan['capacity_videos_per_hour_per_gpu']:,} vidéos/h", f"{plan['throughput_sampled_fps']} FPS analysés")
    m2.metric("VRAM Poids + KV-Cache Libre", f"{plan['weights_vram_gb']} Go / {plan['available_kv_cache_gb']} Go", f"Max {plan['max_concurrent_sequences']} séq. simultanées")
    m3.metric("Latence Effective / Vidéo", f"{plan['effective_latency_per_video_sec']} s", "Continuous Batching vLLM")
    m4.metric("Dimensionnement ASIN (N+1 HA)", f"{plan['recommended_gpus_n_plus_1']} GPU", f"{plan['min_gpus_required']} actif(s) + 1 redondance ({plan['estimated_power_kw']} kW)")

    st.json(plan)

# ==============================================================================
# MODULE 4 : PROTECTION DES DONNÉES SENSIBLES & ANONYMISATION APDP BÉNIN
# ==============================================================================
elif module_choice.startswith("M4"):
    st.subheader("🔒 Module 4 — Anonymisation Vidéo Temps Réel & Conformité Loi n° 2017-20 (APDP Bénin)")
    anon_mode = st.selectbox(
        "Méthode de Masquage des Données à Caractère Personnel (Visages & Plaques) :",
        ["GAUSSIAN_BLUR", "MOSAIC_PIXELATION", "SOLID_CRYPTO_MASK"],
    )
    raw_img, boxes = generate_synthetic_urban_scene()
    anon_img, audit_info = apply_apdp_anonymization(raw_img, boxes, method=anon_mode)

    im_col1, im_col2 = st.columns(2)
    with im_col1:
        st.markdown("**Flux Brut Capturé (Mémoire Volatiles RAM uniquement)**")
        st.image(raw_img, caption="Trame Brute avec 2 Plaques & 2 Visages détectés", use_container_width=True)
    with im_col2:
        st.markdown(f"**Flux Anonymisé APDP ({anon_mode}) — Prêt pour Inférence VLM & Archivage**")
        st.image(anon_img, caption=f"SHA-256 Après Anonymisation : {audit_info['frame_sha256_after']}", use_container_width=True)

    st.markdown("#### 📜 Certificat d'Audit de Conformité APDP & PSSI ASIN")
    st.json(audit_info)

# ==============================================================================
# MODULE 5 : RECHERCHE SÉMANTIQUE VIDÉO & LANGUES NATIONALES
# ==============================================================================
elif module_choice.startswith("M5"):
    st.subheader("🌍 Module 5 — Évolutions Modèle : Recherche Sémantique Multimodale (`SigLIP`) & Multilinguisme")
    q_col1, q_col2 = st.columns([3, 1])
    with q_col1:
        user_query = st.text_input(
            "Requête Sémantique en Langage Naturel sur les Archives Vidéo :",
            value="véhicule immobilisé nuit urgence congestion sécurité",
        )
    with q_col2:
        lang_sel = st.selectbox(
            "Langue de Synthèse VLM :",
            ["Français", "English (Technical)", "Fon (Langue Nationale Bénin)", "Yoruba (Langue Nationale Bénin)"],
        )

    df_search = semantic_video_search(user_query, target_lang=lang_sel)
    st.dataframe(df_search, use_container_width=True)


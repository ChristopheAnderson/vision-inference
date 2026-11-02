# -*- coding: utf-8 -*-
"""
VISIONINFERENCE HUB — PLATEFORME D'ANALYSE VIDÉO VLM & D'OPTIMISATION D'INFÉRENCE GPU (5 MODULES)
Concepteur : Ing. Christophe WAVOEKE (Ingénieur de Conception Bac+5 — Lauréat ENSGMM / UNSTIM)
"""
import os
import sys
import json
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
    page_title="VisionInference Hub — Analyse Vidéo VLM & Optimisation GPU",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light Mode UI & High-Contrast Clean UX
st.markdown(
    """
    <style>
    .stApp { background-color: #f8fafc; color: #0f172a; }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .hero-banner {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 65%, #e0f2fe 100%);
        color: #0f172a;
        padding: 20px 26px;
        border-radius: 12px;
        border: 1px solid #bae6fd;
        border-left: 5px solid #2563eb;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        margin-bottom: 18px;
    }
    .hero-banner h1 { font-size: 24px; font-weight: 800; margin: 0 0 6px 0; color: #0f172a; }
    .hero-banner p { font-size: 13.8px; color: #334155; margin: 0; }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #2563eb;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .kpi-title { font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700; }
    .kpi-value { font-size: 22px; font-weight: 800; color: #0f172a; margin: 2px 0; }
    .kpi-sub { font-size: 11.5px; color: #059669; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "profile.jpg")
    if os.path.exists(profile_path):
        st.image(profile_path, width=110)
    st.markdown("### **Ing. Christophe WAVOEKE**")
    st.caption(
        "Ingénieur IA Vision, Modèles Multimodaux (VLM) & Optimisation d'Inférence GPU\n\n"
        "🎓 **Ingénieur de Conception (Bac+5)** — Génie Mathématique & Informatique (Lauréat ENSGMM / UNSTIM)"
    )
    st.markdown("---")
    st.markdown("#### 🧭 Architecture (5 Modules)")
    st.markdown(
        "- **M1 :** Chaîne Vidéo VLM (`Qwen2.5-VL`), `FFmpeg NVDEC`, Calibration & Consensus\n"
        "- **M2 :** Évaluation (`Précision/Rappel/F1`) & Gate CI/CD de Non-Régression\n"
        "- **M3 :** Optimisation `vLLM` (`AWQ 4-bit` / `FP8`) & Dimensionnement GPU\n"
        "- **M4 :** Anonymisation Temps Réel des Données Personnelles\n"
        "- **M5 :** Recherche Vectorielle Multimodale `SigLIP` & Multilinguisme"
    )
    st.markdown("---")
    st.markdown("🔗 [Code Source GitHub](https://github.com/ChristopheAnderson/vision-inference)")

st.markdown(
    """
    <div class="hero-banner">
        <h1>👁️ VisionInference Hub — Analyse Vidéo VLM & Optimisation d'Inférence GPU</h1>
        <p>Architecture industrielle de traitement vidéo à fort volume : Modèles Vision-Langage à poids ouverts (<b>Qwen2.5-VL-7B / LLaVA-OneVision</b>), Décodage matériel <b>FFmpeg NVDEC</b>, Calibration Probabiliste (<i>Temperature Scaling</i>), Protocole de Non-Régression Déterministe et Serveur d'Inférence <b>vLLM (PagedAttention + AWQ 4-bit)</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        '<div class="kpi-card"><div class="kpi-title">Débit vLLM AWQ 4-Bit</div>'
        '<div class="kpi-value">4 615 vidéos/h</div>'
        '<div class="kpi-sub">Gain ×6.22 vs PyTorch FP16 (1 GPU L4)</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="kpi-card"><div class="kpi-title">Empreinte Mémoire VRAM</div>'
        '<div class="kpi-value">6.4 Go (-65.2%)</div>'
        '<div class="kpi-sub">Contre 18.4 Go en FP16 non quantifié</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="kpi-card"><div class="kpi-title">Calibration & Consensus</div>'
        '<div class="kpi-value">ECE : 0.019</div>'
        '<div class="kpi-sub">Temperature Scaling T*=1.38 · Fenêtre N=5</div></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        '<div class="kpi-card"><div class="kpi-title">Gate Non-Régression</div>'
        '<div class="kpi-value">93.1% F1-Score</div>'
        '<div class="kpi-sub">1 880 vidéos · Tolérance stricte ≤ 1.0%</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

tabs = st.tabs(
    [
        "🎬 M1 : Chaîne VLM Vidéo & Calibration",
        "🛡️ M2 : Évaluation & Non-Régression",
        "⚡ M3 : Optimisation vLLM & Capacité GPU",
        "🔒 M4 : Anonymisation Temps Réel (PII)",
        "🌍 M5 : Recherche Multimodale & Langues",
    ]
)

with tabs[0]:
    st.subheader("Module 1 — Chaîne d'Analyse Vision Documentée, Versionnée et Maintenable")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        duration_s = st.slider("Durée de la séquence vidéo (secondes)", 15, 300, 60, 15)
        fps_target = st.slider("Cadence d'échantillonnage FFmpeg (FPS)", 0.5, 5.0, 2.0, 0.5)
        scene_thresh = st.slider("Seuil de détection de changement de scène (ΔHSV)", 0.10, 0.50, 0.28, 0.02)
        temp_t = st.slider("Température de calibration probabiliste (T*)", 1.00, 2.20, 1.38, 0.02)
        window_n = st.slider("Fenêtre de vérification croisée inter-trames (N)", 3, 9, 5, 2)
    with col_b:
        ffmpeg_cmd = build_ffmpeg_nvdec_command("flux_camera_01.mp4", "keyframes/", fps_target, scene_thresh)
        st.markdown("**Commande d'extraction hybride `FFmpeg NVDEC (CUDA)` générée :**")
        st.code(ffmpeg_cmd, language="bash")
        sim_res = simulate_video_chain_execution(
            video_duration_s=duration_s,
            source_fps=30,
            sampled_fps=fps_target,
            temperature_t=temp_t,
            consensus_window=window_n,
        )
        st.json(sim_res["integrity_and_sampling"])

    df_frames = pd.DataFrame(sim_res["frame_timeline"])
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=df_frames["timestamp_s"],
            y=df_frames["raw_prob"],
            mode="lines+markers",
            name="Probabilité brute VLM (Non calibrée)",
            line=dict(color="#f59e0b", dash="dot"),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=df_frames["timestamp_s"],
            y=df_frames["calibrated_prob"],
            mode="lines+markers",
            name=f"Probabilité calibrée (T*={temp_t}) + Consensus N={window_n}",
            line=dict(color="#2563eb", width=3),
        )
    )
    fig1.add_hline(y=0.72, line_dash="dash", line_color="#059669", annotation_text="Seuil Opérationnel Calibré (0.72)")
    fig1.update_layout(
        title="Calibration Temperature Scaling & Filtrage des Faux Positifs Isolés par Vérification Croisée",
        xaxis_title="Temps Vidéo (secondes)",
        yaxis_title="Probabilité de Détection",
        template="plotly_white",
        height=390,
    )
    st.plotly_chart(fig1, use_container_width=True)

with tabs[1]:
    st.subheader("Module 2 — Méthode d'Évaluation Opérationnelle & Protocole de Non-Régression")
    max_drop = st.slider("Tolérance maximale de baisse de F1-Score autorisée (Gate CI/CD)", 0.002, 0.020, 0.010, 0.001)
    eval_report = run_non_regression_protocol(max_allowed_f1_drop=max_drop)
    if eval_report["overall_gate_passed"]:
        st.success(
            f"✅ GATE DE NON-RÉGRESSION VALIDÉE — Manifeste SHA-256 : `{eval_report['reproducibility_manifest_sha256'][:24]}...`"
        )
    else:
        st.error("❌ GATE DE NON-RÉGRESSION BLOQUÉE — Déploiement refusé.")
    st.dataframe(pd.DataFrame(eval_report["categories_ detail"]), use_container_width=True)

with tabs[2]:
    st.subheader("Module 3 — Gains d'Optimisation Mesurés (`vLLM` / Quantification) & Capacité par GPU")
    df_bench = get_optimization_benchmarks_df()
    st.dataframe(df_bench, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        gpu_choice = st.selectbox("Sélectionner l'unité de calcul GPU de référence", list(GPU_HARDWARE_SPECS.keys()), index=1)
        daily_target = st.number_input("Volume cible de vidéos à analyser par 24h", 1000, 1000000, 100000, 5000)
        avg_dur = st.slider("Durée moyenne par vidéo (s)", 15, 180, 60, 15)
        cap_plan = compute_gpu_capacity_plan(gpu_choice, avg_dur, 2.0, 8, daily_target)
        st.info(
            f"**Capacité par unité `{gpu_choice}` :** `{cap_plan['Reference_Videos_Per_Hour_1_GPU']:,}` vidéos/heure "
            f"(`{cap_plan['Reference_Videos_Per_24h_1_GPU']:,}` vidéos/24h)\n\n"
            f"**Dimensionnement recommandé pour `{daily_target:,}` vidéos/jour :** "
            f"**`{cap_plan['Required_GPU_Units_With_15pct_Headroom']} unité(s) GPU`** (marge de sécurité 15% incluse)."
        )
    with c_right:
        fig_bench = px.bar(
            df_bench,
            x="Configuration",
            y="Throughput_Videos_Per_Hour",
            color="VRAM_Peak_GB",
            text="Speedup_Factor",
            title="Capacité Horaire (Vidéos/h) & Empreinte VRAM Pic (Go)",
            template="plotly_white",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_bench, use_container_width=True)

with tabs[3]:
    st.subheader("Module 4 — Anonymisation Temps Réel Visages & Plaques (Loi N° 2017-20 / RGPD)")
    raw_img, boxes = generate_synthetic_urban_scene()
    anon_img, audit = apply_apdp_anonymization(raw_img, boxes)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Trame Brute Capturée (Données personnelles exposées)**")
        st.image(raw_img, use_container_width=True)
    with col2:
        st.markdown("**Trame Anonymisée à la Volée en RAM (Conforme Protection des Données)**")
        st.image(anon_img, use_container_width=True)
    st.json(audit)

with tabs[4]:
    st.subheader("Module 5 — Recherche Multimodale Vectorielle (`SigLIP`) & Synthèses Multilingues")
    query_txt = st.text_input("Requête sémantique multimodale :", "Intrusion périmétrique nocturne détectée près du dépôt")
    lang_code = st.selectbox("Langue de synthèse :", ["fr", "en", "fon", "yo"], index=0)
    results = semantic_video_search(query_txt, top_k=3, target_lang=lang_code)
    st.dataframe(pd.DataFrame(results), use_container_width=True)

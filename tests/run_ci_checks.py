# -*- coding: utf-8 -*-
"""
Suite de tests automatisés CI/CD — VisionEval Non-Regression Gate
Exécutée par GitHub Actions à chaque push sur la branche main.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.module1_vlm_video_chain import simulate_video_chain_execution
from src.module2_eval_non_regression import run_non_regression_protocol
from src.module3_vllm_capacity_optimizer import compute_gpu_capacity_plan


def main() -> None:
    # 1. Test Module 1 : Chaîne VLM & Calibration
    _, summary_m1 = simulate_video_chain_execution()
    assert summary_m1["compression_ratio_pct"] > 85.0, "Taux d'élagage FFmpeg insuffisant"
    print("[PASS] Module 1 — Chaîne Vidéo VLM & Calibration :", summary_m1)

    # 2. Test Module 2 : Gate de Non-Régression sur 1 880 vidéos
    _, manifest_m2 = run_non_regression_protocol(max_allowed_f1_drop_pct=1.0)
    assert manifest_m2["gate_decision"] == "APPROVED_FOR_PRODUCTION", "Échec de la Gate de Non-Régression"
    print("[PASS] Module 2 — Non-Regression Gate SHA-256 :", manifest_m2["reproducibility_sha256"])

    # 3. Test Module 3 : Dimensionnement Capacitaire GPU
    cap_m3 = compute_gpu_capacity_plan("NVIDIA L4 (24 Go VRAM)", 100000, 60, 8)
    assert cap_m3["capacity_videos_per_hour_per_gpu"] > 3000, "Capacité GPU L4 sous le seuil nominal"
    print("[PASS] Module 3 — Capacity Planner GPU :", cap_m3["capacity_videos_per_hour_per_gpu"], "vidéos/h")


if __name__ == "__main__":
    main()


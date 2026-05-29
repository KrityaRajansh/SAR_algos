"""
run_all_experiments.py — Common Experiment Runner
==================================================
Runs all four SAR algorithms on *exactly the same* simulated data,
collects metrics and images, and produces:

  sar_results/
    FA-PnP-ADMM_result.png
    FA-CNN-ADMM_result.png
    FA-AE-ADMM_result.png
    RMA_result.png
    SUMMARY_comparison.png    ← 4-panel side-by-side
    SUMMARY_metrics.png       ← grouped bar chart
    SUMMARY_table.csv         ← CSV for copy-paste into slides

Experiments performed
---------------------
  E1  Single scene, single SNR (15 dB) — primary comparison
  E2  SNR sweep [5, 10, 15, 20, 25 dB] — robustness to noise
  E3  Convergence speed comparison (ADMM algorithms only)
  E4  Image-quality radar chart (PSNR, SSIM, ENL, Contrast, Speed)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time, os, csv

from sar_utils import (
    get_shared_data, simulate_raw_signal, make_reference,
    compute_metrics, save_comparison_figure, save_summary_comparison,
    SAR_PARAMS, make_scene,
)

# ── Import algorithm functions ────────────────────────────────────────────────
from algo1_FA_PnP_ADMM import fa_pnp_admm
from algo2_FA_CNN_ADMM  import fa_cnn_admm
from algo3_FA_AE_ADMM   import fa_ae_admm
from algo4_RMA          import rma

OUTPUT_DIR = "sar_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1 — Primary comparison (same scene, SNR = 15 dB)
# ════════════════════════════════════════════════════════════════════════
def experiment_1():
    print("\n" + "═"*60)
    print("  EXPERIMENT 1 — Primary comparison  (SNR = 15 dB)")
    print("═"*60)

    scene, raw, ref, params = get_shared_data()
    results = []

    configs = [
        ("FA-PnP-ADMM", lambda r, p: fa_pnp_admm(r, p)),
        ("FA-CNN-ADMM",  lambda r, p: fa_cnn_admm(r, p)),
        ("FA-AE-ADMM",   lambda r, p: fa_ae_admm(r, p)),
        ("RMA",          lambda r, p: (rma(r, p), None)),
    ]

    all_results = []
    for name, fn in configs:
        print(f"\n  ▶  {name}")
        t0 = time.perf_counter()
        out = fn(raw, params)
        rt  = time.perf_counter() - t0
        recon = out[0]
        conv  = out[1]

        m = compute_metrics(recon, ref)
        print(f"     PSNR={m['PSNR']:.2f} dB  SSIM={m['SSIM']:.4f}  "
              f"ENL={m['ENL']:.2f}  Contrast={m['Contrast']:.2f}  "
              f"time={rt:.3f}s")

        save_comparison_figure(recon, ref, name, m, rt, OUTPUT_DIR, conv)
        all_results.append(dict(name=name, recon=recon, metrics=m, runtime=rt))

    save_summary_comparison(all_results, OUTPUT_DIR)

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, "SUMMARY_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm","PSNR (dB)","SSIM","ENL","Contrast","Runtime (s)"])
        for r in all_results:
            writer.writerow([
                r["name"],
                f"{r['metrics']['PSNR']:.3f}",
                f"{r['metrics']['SSIM']:.4f}",
                f"{r['metrics']['ENL']:.2f}",
                f"{r['metrics']['Contrast']:.2f}",
                f"{r['runtime']:.3f}",
            ])
    print(f"\n  [saved] {csv_path}")
    return all_results


# ════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2 — SNR sweep
# ════════════════════════════════════════════════════════════════════════
def experiment_2():
    print("\n" + "═"*60)
    print("  EXPERIMENT 2 — SNR sweep  [5, 10, 15, 20, 25 dB]")
    print("═"*60)

    snr_levels = [5, 10, 15, 20, 25]
    params     = dict(SAR_PARAMS)
    scene      = make_scene(params["Nr"], params["Na"])
    ref        = make_reference(scene)

    algos = {
        "FA-PnP-ADMM": lambda r, p: fa_pnp_admm(r, p)[0],
        "FA-CNN-ADMM":  lambda r, p: fa_cnn_admm(r, p)[0],
        "FA-AE-ADMM":   lambda r, p: fa_ae_admm(r, p)[0],
        "RMA":          lambda r, p: rma(r, p),
    }
    palette = {"FA-PnP-ADMM": "#00e5ff", "FA-CNN-ADMM": "#ff4081",
               "FA-AE-ADMM":  "#69f0ae", "RMA":         "#ffab40"}

    # Store PSNR and SSIM per algo per SNR
    data = {n: {"psnr": [], "ssim": []} for n in algos}

    for snr in snr_levels:
        params["SNR_dB"] = snr
        raw, _ = simulate_raw_signal(scene, params)
        print(f"  SNR={snr:2d} dB", end="")
        for name, fn in algos.items():
            recon = fn(raw, params)
            m     = compute_metrics(recon, ref)
            data[name]["psnr"].append(m["PSNR"])
            data[name]["ssim"].append(m["SSIM"])
            print(f"  {name}: {m['PSNR']:.1f}dB", end="")
        print()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d0d0d")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444444")

    for name in algos:
        c = palette[name]
        ax1.plot(snr_levels, data[name]["psnr"], "o-", color=c, label=name, lw=2)
        ax2.plot(snr_levels, data[name]["ssim"], "s--", color=c, label=name, lw=2)

    ax1.set_xlabel("SNR (dB)", color="white"); ax1.set_ylabel("PSNR (dB)", color="white")
    ax1.set_title("PSNR vs SNR", color="white"); ax1.legend(facecolor="#222222", labelcolor="white")
    ax2.set_xlabel("SNR (dB)", color="white"); ax2.set_ylabel("SSIM", color="white")
    ax2.set_title("SSIM vs SNR", color="white"); ax2.legend(facecolor="#222222", labelcolor="white")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "E2_snr_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"\n  [saved] {path}")


# ════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 3 — Convergence comparison (ADMM algos)
# ════════════════════════════════════════════════════════════════════════
def experiment_3(e1_results):
    """
    Plots iteration-vs-residual for the three ADMM variants on the same axes.
    RMA is excluded (single-pass).
    """
    print("\n" + "═"*60)
    print("  EXPERIMENT 3 — Convergence comparison (ADMM)")
    print("═"*60)

    scene, raw, ref, params = get_shared_data()
    admm_configs = [
        ("FA-PnP-ADMM", lambda r, p: fa_pnp_admm(r, p, max_iter=100)),
        ("FA-CNN-ADMM",  lambda r, p: fa_cnn_admm(r, p,  max_iter=80)),
        ("FA-AE-ADMM",   lambda r, p: fa_ae_admm(r, p,   max_iter=70)),
    ]
    palette = {"FA-PnP-ADMM": "#00e5ff", "FA-CNN-ADMM": "#ff4081",
               "FA-AE-ADMM":  "#69f0ae"}

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#111111"); ax.tick_params(colors="white")
    ax.spines[:].set_color("#444444")

    for name, fn in admm_configs:
        _, residuals = fn(raw, params)
        c = palette[name]
        ax.semilogy(residuals, color=c, label=name, lw=2)
        ax.axhline(residuals[-1], color=c, lw=0.5, ls="--", alpha=0.5)

    ax.set_xlabel("Iteration", color="white")
    ax.set_ylabel("Normalised primal residual (log)", color="white")
    ax.set_title("Convergence: FA-PnP / FA-CNN / FA-AE ADMM", color="white")
    ax.legend(facecolor="#222222", labelcolor="white")

    path = os.path.join(OUTPUT_DIR, "E3_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  [saved] {path}")


# ════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 4 — Radar chart
# ════════════════════════════════════════════════════════════════════════
def experiment_4(all_results):
    """
    Spider / radar chart comparing normalised metrics across all 4 algos.
    """
    print("\n" + "═"*60)
    print("  EXPERIMENT 4 — Radar (spider) chart")
    print("═"*60)

    categories = ["PSNR", "SSIM", "ENL", "Contrast", "Speed"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    palette = ["#00e5ff", "#ff4081", "#69f0ae", "#ffab40"]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True),
                           facecolor="#0d0d0d")
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("#444444")

    max_rt = max(r["runtime"] for r in all_results)

    for res, color in zip(all_results, palette):
        m = res["metrics"]
        vals = [
            m["PSNR"] / 50,
            m["SSIM"],
            min(m["ENL"] / 20, 1.0),
            min(m["Contrast"] / 10, 1.0),
            1.0 - res["runtime"] / max_rt,   # higher speed = lower time
        ]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", color=color, lw=2, label=res["name"])
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="white", fontsize=11)
    ax.set_yticklabels([]); ax.set_ylim(0, 1)
    ax.set_title("Algorithm Radar Chart (normalised)", color="white",
                 pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor="#222222", labelcolor="white")

    path = os.path.join(OUTPUT_DIR, "E4_radar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  [saved] {path}")


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n  SAR Algorithm Comparison Framework")
    print("  Running all 4 experiments …\n")

    e1 = experiment_1()
    experiment_2()
    experiment_3(e1)
    experiment_4(e1)

    print("\n" + "═"*60)
    print("  All experiments complete.")
    print(f"  Results saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("═"*60)
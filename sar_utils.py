"""
sar_utils.py — Shared SAR simulation framework
================================================
Uses a STRUCTURED reflectivity scene (terrain + buildings + roads +
point targets) so reconstructed images actually look like SAR imagery.

Forward model
-------------
  y = PSF * x + speckle + AWGN

where PSF combines:
  • range direction : sinc (finite bandwidth)
  • azimuth direction: Gaussian (partial aperture / defocus)

The algorithms solve the inverse problem of recovering x from y,
giving visually meaningful and *different* reconstructions.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import windows
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
RNG  = np.random.default_rng(SEED)

# ── Common parameters ────────────────────────────────────────────────────────
SAR_PARAMS = dict(
    Nr               = 256,
    Na               = 256,
    SNR_dB           = 15.0,
    psf_range_width  = 3.0,   # sinc main-lobe half-width (pixels)
    psf_az_sigma     = 2.5,   # azimuth Gaussian sigma (pixels)
)


# ════════════════════════════════════════════════════════════════════
# SCENE — structured, visually meaningful SAR reflectivity map
# ════════════════════════════════════════════════════════════════════
def make_scene(Nr, Na):
    """
    Layers:
      1. Smooth terrain background
      2. Urban block grid  (bright rectangles)
      3. Roads             (thin bright lines)
      4. Forest patches    (medium-intensity textured blobs)
      5. Water body        (very dark smooth region)
      6. Corner reflectors (sharp bright point targets)
    """
    scene = np.zeros((Nr, Na), dtype=np.float64)

    # 1. Terrain background
    bg = RNG.standard_normal((Nr // 8, Na // 8))
    bg_up = np.repeat(np.repeat(bg, 8, axis=0), 8, axis=1)[:Nr, :Na]
    terrain = gaussian_filter(bg_up, sigma=6)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-12)
    scene += 0.25 * terrain

    # 2. Urban blocks
    for (r0, c0) in [(30,30),(30,140),(80,80),(140,30),(140,150),
                     (170,90),(60,190),(190,190),(50,110),(110,50)]:
        h = int(RNG.integers(12, 25))
        w = int(RNG.integers(12, 25))
        r1, c1 = min(r0+h, Nr), min(c0+w, Na)
        intensity = float(RNG.uniform(0.6, 1.0))
        patch = intensity + 0.15 * RNG.standard_normal((r1-r0, c1-c0))
        scene[r0:r1, c0:c1] += np.clip(patch, 0, 1)

    # 3. Roads
    scene[120:124, 10:240] += 0.55
    scene[10:240, 118:122] += 0.50
    for k in range(180):
        r = int(20 + k * 0.8); c = int(20 + k)
        if 0 <= r < Nr-2 and 0 <= c < Na-2:
            scene[r:r+3, c:c+3] += 0.45

    # 4. Forest patches
    rr, cc = np.ogrid[:Nr, :Na]
    for (cr, ccr, rad) in [(70, 70, 35), (185, 160, 28)]:
        mask = np.exp(-((rr-cr)**2 + (cc-ccr)**2) / (2*rad**2))
        texture = RNG.standard_normal((Nr, Na)) * 0.12 + 0.35
        scene += mask * np.clip(texture, 0, 1)

    # 5. Water body (dark)
    water = np.exp(-((rr-210)**2/900 + (cc-50)**2/600))
    scene -= 0.4 * water

    # 6. Point targets
    for (tr, tc, amp) in [
        (45,50,1.0),(45,160,0.95),(128,128,1.0),(200,40,0.9),
        (200,200,0.85),(100,210,0.8),(160,100,0.75),(75,100,0.7),(230,130,0.65)
    ]:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r2 = int(np.clip(tr+dr, 0, Nr-1))
                c2 = int(np.clip(tc+dc, 0, Na-1))
                scene[r2, c2] += amp * np.exp(-(dr**2+dc**2)/0.8)

    scene = np.clip(scene, 0, None)
    scene /= scene.max() + 1e-12
    return scene


# ════════════════════════════════════════════════════════════════════
# PSF
# ════════════════════════════════════════════════════════════════════
def make_psf(params, ksz=21):
    half   = ksz // 2
    idx    = np.arange(-half, half+1, dtype=np.float64)
    sinc_r = np.sinc(idx / params["psf_range_width"])
    ham_r  = windows.hamming(ksz)
    az_ker = np.exp(-idx**2 / (2*params["psf_az_sigma"]**2))
    az_ker /= az_ker.sum()
    psf    = np.outer(sinc_r * ham_r, az_ker)
    psf   /= psf.sum()
    return psf


# ════════════════════════════════════════════════════════════════════
# FORWARD MODEL
# ════════════════════════════════════════════════════════════════════
def simulate_raw_signal(scene, params):
    psf   = make_psf(params)
    clean = convolve(scene.astype(np.float64), psf, mode="reflect")
    speckle = RNG.exponential(1.0, scene.shape)
    blurred = clean * speckle
    sig_pwr = np.mean(blurred**2)
    noise_s = np.sqrt(sig_pwr / (10**(params["SNR_dB"]/10)))
    noise   = RNG.normal(0, noise_s, scene.shape)
    return blurred + noise, clean


def make_reference(scene):
    ref = np.abs(scene).astype(np.float64)
    return ref / (ref.max() + 1e-12)


# ════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════
def compute_metrics(recon, reference):
    def norm(x):
        x = np.abs(x).astype(np.float64)
        return x / (x.max() + 1e-12)
    r = norm(recon); g = norm(reference)
    p   = psnr(g, r, data_range=1.0)
    s   = ssim(g, r, data_range=1.0)
    enl = (np.mean(r)**2) / (np.var(r) + 1e-12)
    thr = np.percentile(r, 90)
    con = np.mean(r[r>thr]) / (np.mean(r[r<=thr]) + 1e-12)
    return dict(PSNR=p, SSIM=s, ENL=enl, Contrast=con)


# ════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════
CMAP = "gray"

def _n(img):
    a = np.abs(img).astype(np.float64)
    return a / (a.max() + 1e-12)


def save_comparison_figure(recon_img, reference_img, algo_name,
                           metrics, runtime, out_dir=".", convergence=None):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(16, 9), facecolor="#0d0d0d")
    fig.suptitle(
        f"{algo_name}  —  PSNR {metrics['PSNR']:.2f} dB  |  "
        f"SSIM {metrics['SSIM']:.4f}  |  Runtime {runtime:.2f}s",
        color="white", fontsize=14, fontweight="bold", y=0.97)

    ref_n = _n(reference_img); rec_n = _n(recon_img)
    diff  = np.abs(ref_n - rec_n)
    ax_kw = dict(facecolor="#111111")

    ax1 = fig.add_subplot(2,3,1,**ax_kw)
    ax1.imshow(ref_n, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax1.set_title("Reference (clean scene)", color="white"); ax1.axis("off")

    ax2 = fig.add_subplot(2,3,2,**ax_kw)
    ax2.imshow(rec_n, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax2.set_title(f"Reconstructed — {algo_name}", color="white"); ax2.axis("off")

    ax3 = fig.add_subplot(2,3,3,**ax_kw)
    im3 = ax3.imshow(diff, cmap="hot", vmin=0, vmax=0.5, aspect="auto")
    ax3.set_title("Error map", color="white"); ax3.axis("off")
    cb = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    ax4 = fig.add_subplot(2,3,4,**ax_kw)
    if convergence and len(convergence) > 1:
        ax4.semilogy(convergence, color="#00e5ff", lw=2)
        ax4.set_xlabel("Iteration", color="white")
        ax4.set_ylabel("Residual (log)", color="white")
        ax4.set_title("Convergence", color="white")
    else:
        ax4.text(0.5, 0.5, "Single-pass\n(no iterations)",
                 ha="center", va="center", color="#888888", fontsize=12)
        ax4.set_title("Convergence", color="white")
    ax4.tick_params(colors="white"); ax4.spines[:].set_color("#444444")

    ax5 = fig.add_subplot(2,3,5,**ax_kw)
    keys   = ["PSNR\n(norm)","SSIM","ENL\n(norm)","Contrast\n(norm)"]
    nm     = [metrics["PSNR"]/50, metrics["SSIM"],
              min(metrics["ENL"]/20,1.0), min(metrics["Contrast"]/10,1.0)]
    cols   = ["#00e5ff","#ff4081","#69f0ae","#ffab40"]
    bars   = ax5.bar(range(4), nm, color=cols, edgecolor="#222222")
    ax5.set_ylim(0, 1.15); ax5.set_title("Normalised Metrics", color="white")
    ax5.set_xticks(range(4)); ax5.set_xticklabels(keys, color="white", fontsize=8)
    ax5.tick_params(colors="white"); ax5.spines[:].set_color("#444444")
    for bar, val in zip(bars, [metrics["PSNR"],metrics["SSIM"],
                                metrics["ENL"],metrics["Contrast"]]):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f"{val:.2f}", ha="center", color="white", fontsize=8)

    ax6 = fig.add_subplot(2,3,6,**ax_kw); ax6.axis("off")
    for i, ln in enumerate([
        f"Algorithm :  {algo_name}",
        f"PSNR      :  {metrics['PSNR']:.3f} dB",
        f"SSIM      :  {metrics['SSIM']:.4f}",
        f"ENL       :  {metrics['ENL']:.2f}",
        f"Contrast  :  {metrics['Contrast']:.2f}",
        f"Runtime   :  {runtime:.3f} s",
    ]):
        ax6.text(0.05, 0.85-i*0.13, ln, color="white", fontsize=11,
                 fontfamily="monospace", transform=ax6.transAxes)
    ax6.set_title("Summary", color="white")

    plt.tight_layout(rect=[0,0,1,0.95])
    path = os.path.join(out_dir, f"{algo_name.replace(' ','_')}_result.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


def save_summary_comparison(results, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), facecolor="#0d0d0d")
    fig.suptitle("SAR Reconstruction — All Algorithms",
                 color="white", fontsize=14, fontweight="bold")
    for ax, res in zip(axes, results):
        ax.set_facecolor("#111111")
        ax.imshow(_n(res["recon"]), cmap=CMAP, vmin=0, vmax=1, aspect="auto")
        ax.set_title(
            f"{res['name']}\nPSNR {res['metrics']['PSNR']:.1f} dB  "
            f"SSIM {res['metrics']['SSIM']:.3f}\n{res['runtime']:.2f}s",
            color="white", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, "SUMMARY_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig); print(f"  [saved] {path}")

    fig2, ax = plt.subplots(figsize=(10, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#111111")
    mk = ["PSNR","SSIM","ENL","Contrast","Speed"]
    x  = np.arange(len(mk)); w = 0.8/n
    pal= ["#00e5ff","#ff4081","#69f0ae","#ffab40"]
    mrt= max(r["runtime"] for r in results)
    for i, res in enumerate(results):
        m = res["metrics"]
        v = [m["PSNR"]/50, m["SSIM"], min(m["ENL"]/20,1),
             min(m["Contrast"]/10,1), 1-res["runtime"]/mrt]
        ax.bar(x+i*w-0.4+w/2, v, w, label=res["name"],
               color=pal[i%len(pal)], edgecolor="#222222")
    ax.set_xticks(x); ax.set_xticklabels(mk, color="white")
    ax.set_ylabel("Normalised value", color="white")
    ax.set_title("Metric Comparison (normalised)", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444444")
    ax.legend(facecolor="#222222", labelcolor="white")
    plt.tight_layout()
    path2 = os.path.join(out_dir, "SUMMARY_metrics.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig2); print(f"  [saved] {path2}")
    return path, path2


def get_shared_data():
    p     = SAR_PARAMS
    scene = make_scene(p["Nr"], p["Na"])
    raw,_ = simulate_raw_signal(scene, p)
    ref   = make_reference(scene)
    return scene, raw, ref, p
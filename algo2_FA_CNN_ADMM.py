"""
algo2_FA_CNN_ADMM.py — Fast & Adaptive CNN Plug-and-Play ADMM
=============================================================
Denoiser : 5-layer DnCNN-style network in pure NumPy.
           Fixed random weights (deep-image-prior / untrained network).
           Provides more spatially-structured denoising than soft-thresh.

Same ADMM skeleton as algo1; only the z-update differs.
"""

import numpy as np
from scipy.ndimage import convolve, correlate
import time, sys, os

sys.path.insert(0, os.path.dirname(__file__))
from sar_utils import get_shared_data, compute_metrics, save_comparison_figure, make_psf

OUTPUT_DIR = "sar_results"


# ── Minimal CNN (pure NumPy, real-valued) ────────────────────────────────────
class DnCNN_lite:
    """
    5-layer conv net operating on 2-D real images.
    Channels: 1→16→32→16→8→1
    All 3×3 kernels except final 1×1.
    Fixed initialisation — acts as structured prior.
    """
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        def he(out, inp, k):
            return rng.normal(0, np.sqrt(2.0/(inp*k*k)),
                              (out, inp, k, k)).astype(np.float64)
        self.W = [he(16,1,3), he(32,16,3), he(16,32,3),
                  he(8,16,3), he(1,8,1)]
        self.b = [np.zeros(w.shape[0]) for w in self.W]

    def _conv(self, feat, W, b):
        out_ch, in_ch, kH, kW = W.shape
        H, HW = feat.shape[1], feat.shape[2]
        out = np.zeros((out_ch, H, HW))
        for o in range(out_ch):
            for i in range(in_ch):
                pad = kH // 2
                fp  = np.pad(feat[i], pad, mode="reflect")
                # 2-D correlation via sliding window
                for r in range(kH):
                    for c in range(kW):
                        out[o] += W[o,i,r,c] * fp[r:r+H, c:c+HW]
            out[o] += b[o]
        return out

    def forward(self, x2d):
        feat = x2d[np.newaxis]                     # (1,H,W)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            feat = self._conv(feat, W, b)
            if i < len(self.W)-1:
                feat = np.maximum(feat, 0.01*feat)  # leaky ReLU
        return feat[0]


_CNN = None  # singleton — build once

def cnn_denoiser(z, noise_level=0.05):
    global _CNN
    if _CNN is None:
        _CNN = DnCNN_lite(seed=42)
    s   = np.abs(z).max() + 1e-12
    zn  = z / s
    dn  = _CNN.forward(zn)
    # residual learning: output is noise estimate → subtract
    return np.clip(zn - dn, 0, None) * s


# ── Shared linear ops ────────────────────────────────────────────────────────
def apply_H(x, psf):  return convolve(x, psf, mode="reflect")
def apply_HT(y, psf): return convolve(y, np.flip(psf), mode="reflect")

def cg_solve(psf, rho, rhs, x0, n_iter=6):
    x = x0.copy()
    r = rhs - (apply_HT(apply_H(x,psf),psf) + rho*x)
    p = r.copy(); rs = np.dot(r.ravel(), r.ravel())
    for _ in range(n_iter):
        Ap = apply_HT(apply_H(p,psf),psf) + rho*p
        a  = rs / (np.dot(p.ravel(), Ap.ravel()) + 1e-12)
        x += a*p; r -= a*Ap
        rn = np.dot(r.ravel(), r.ravel())
        if rn < 1e-20: break
        p = r + (rn/rs)*p; rs = rn
    return x


def fa_cnn_admm(raw, params, rho=0.1, max_iter=60,
                tol=1e-4, adapt_freq=10, mu=2.0, tau=10.0):
    psf  = make_psf(params)
    HT_y = apply_HT(raw, psf)
    x = HT_y.copy(); z = x.copy(); u = np.zeros_like(x)
    residuals = []

    for k in range(max_iter):
        rhs   = HT_y + rho*(z - u)
        x_new = cg_solve(psf, rho, rhs, x)
        z_new = cnn_denoiser(x_new + u)
        u_new = u + x_new - z_new

        pr = np.linalg.norm(x_new-z_new)/(np.linalg.norm(x_new)+1e-12)
        dr = rho*np.linalg.norm(z_new-z)/(np.linalg.norm(u_new)+1e-12)
        residuals.append(float(pr))

        if (k+1) % adapt_freq == 0:
            if pr > tau*dr: rho *= mu
            elif dr > tau*pr: rho /= mu

        x, z, u = x_new, z_new, u_new
        if k % 10 == 0: print(f"    iter {k:3d}  res={pr:.4e}")
        if pr < tol:
            print(f"    Converged at iter {k+1}"); break

    return np.clip(x, 0, None), residuals


if __name__ == "__main__":
    print("="*60); print("  FA-CNN-ADMM  (DnCNN-lite denoiser)"); print("="*60)
    scene, raw, ref, params = get_shared_data()
    t0 = time.perf_counter()
    recon, res = fa_cnn_admm(raw, params)
    rt = time.perf_counter() - t0
    m  = compute_metrics(recon, ref)
    print(f"  Done {rt:.2f}s  PSNR={m['PSNR']:.2f}dB  SSIM={m['SSIM']:.4f}")
    save_comparison_figure(recon, ref, "FA-CNN-ADMM", m, rt, OUTPUT_DIR, res)
"""
algo3_FA_AE_ADMM.py — Fast & Adaptive Autoencoder Plug-and-Play ADMM
=====================================================================
Denoiser : shallow patch autoencoder with SKIP CONNECTION.
           AE predicts a residual correction; output = input - residual.
           This guarantees the AE can never blank the image, only
           smooth / denoise it.

The bottleneck (16 units for 8x8=64 dim patches) forces the AE to
learn a low-rank approximation of local patch structure → produces
smoother, lower-speckle images at the cost of some sharpness.
"""

import numpy as np
from scipy.ndimage import convolve
import time, sys, os

sys.path.insert(0, os.path.dirname(__file__))
from sar_utils import get_shared_data, compute_metrics, save_comparison_figure, make_psf

OUTPUT_DIR = "sar_results"


class PatchAE:
    """
    Residual patch autoencoder.
    Input patch x → encoder → bottleneck → decoder → residual r
    Output = x - alpha * r   (skip connection, alpha=0.4)
    """
    def __init__(self, patch=8, bottleneck=16, seed=42):
        rng  = np.random.default_rng(seed)
        D    = patch * patch
        sc   = np.sqrt(1.0 / D)
        self.P     = patch
        self.alpha = 0.4          # residual blend weight
        self.We = rng.normal(0, sc, (bottleneck, D)).astype(np.float64)
        self.be = np.zeros(bottleneck)
        self.Wd = rng.normal(0, sc, (D, bottleneck)).astype(np.float64)
        self.bd = np.zeros(D)

    def _fwd_patch(self, v):
        h   = np.maximum(0, self.We @ v + self.be)          # ReLU
        res = np.tanh(self.Wd @ h + self.bd) * 0.5          # residual in [-0.5,0.5]
        return np.clip(v - self.alpha * res, 0, None)        # skip + clip

    def forward(self, img):
        P = self.P; H, W = img.shape
        s   = img.max() + 1e-12
        img = img / s
        out = np.zeros_like(img); cnt = np.zeros_like(img)
        step = P // 2
        for r in range(0, H - P + 1, step):
            for c in range(0, W - P + 1, step):
                patch = img[r:r+P, c:c+P].ravel().copy()
                dn    = self._fwd_patch(patch).reshape(P, P)
                out[r:r+P, c:c+P] += dn
                cnt[r:r+P, c:c+P] += 1
        cnt = np.where(cnt == 0, 1, cnt)
        return np.clip(out / cnt, 0, 1) * s


_AE = None

def ae_denoiser(z):
    global _AE
    if _AE is None:
        _AE = PatchAE(patch=8, bottleneck=16, seed=42)
    return _AE.forward(np.clip(z, 0, None))


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


def fa_ae_admm(raw, params, rho=0.1, max_iter=50,
               tol=1e-4, adapt_freq=10, mu=2.0, tau=10.0):
    psf  = make_psf(params)
    HT_y = apply_HT(raw, psf)
    x = HT_y.copy(); z = x.copy(); u = np.zeros_like(x)
    residuals = []

    for k in range(max_iter):
        rhs   = HT_y + rho*(z - u)
        x_new = cg_solve(psf, rho, rhs, x)
        z_new = ae_denoiser(x_new + u)
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
    print("="*60); print("  FA-AE-ADMM  (residual patch autoencoder)"); print("="*60)
    scene, raw, ref, params = get_shared_data()
    t0 = time.perf_counter()
    recon, res = fa_ae_admm(raw, params)
    rt = time.perf_counter() - t0
    m  = compute_metrics(recon, ref)
    print(f"  Done {rt:.2f}s  PSNR={m['PSNR']:.2f}dB  SSIM={m['SSIM']:.4f}")
    save_comparison_figure(recon, ref, "FA-AE-ADMM", m, rt, OUTPUT_DIR, res)
"""
algo1_FA_PnP_ADMM.py — Fast & Adaptive Plug-and-Play ADMM
==========================================================
Denoiser : soft-thresholding (wavelet-like shrinkage).

Forward model:  y = Hx + n   (H = convolution with SAR PSF)
ADMM solves:    min (1/2)||Hx-y||^2 + λ R(x)

Iterates
--------
  x-update : x = (HᵀH + ρI)^{-1} (Hᵀy + ρ(z - u))
              solved via conjugate-gradient (3 iterations, cheap)
  z-update : z = soft_threshold(x + u,  λ/ρ)
  u-update : u = u + x - z

Adaptive ρ: scaled up/down every `adapt_freq` iterations based on
primal/dual residual ratio (Boyd et al., 2011).
"""

import numpy as np
from scipy.ndimage import convolve
import time, sys, os

sys.path.insert(0, os.path.dirname(__file__))
from sar_utils import get_shared_data, compute_metrics, save_comparison_figure, make_psf

OUTPUT_DIR = "sar_results"


def soft_threshold(z, thresh):
    return np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)


def apply_H(x, psf):
    return convolve(x, psf, mode="reflect")

def apply_HT(y, psf):
    return convolve(y, np.flip(psf), mode="reflect")

def apply_HTH_plus_rhoI(x, psf, rho):
    return apply_HT(apply_H(x, psf), psf) + rho * x


def cg_solve(psf, rho, rhs, x0, n_iter=6):
    """Conjugate gradient for (HᵀH + ρI)x = rhs."""
    x = x0.copy()
    r = rhs - apply_HTH_plus_rhoI(x, psf, rho)
    p = r.copy(); rsold = np.dot(r.ravel(), r.ravel())
    for _ in range(n_iter):
        Ap    = apply_HTH_plus_rhoI(p, psf, rho)
        alpha = rsold / (np.dot(p.ravel(), Ap.ravel()) + 1e-12)
        x    += alpha * p
        r    -= alpha * Ap
        rsnew = np.dot(r.ravel(), r.ravel())
        if rsnew < 1e-20: break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


def fa_pnp_admm(raw, params, lam=0.04, rho=0.1, max_iter=80,
                tol=1e-4, adapt_freq=10, mu=2.0, tau=10.0):
    psf = make_psf(params)
    HT_y = apply_HT(raw, psf)
    x = HT_y.copy(); z = x.copy(); u = np.zeros_like(x)
    residuals = []

    for k in range(max_iter):
        rhs   = HT_y + rho * (z - u)
        x_new = cg_solve(psf, rho, rhs, x)
        z_new = soft_threshold(x_new + u, lam / rho)
        u_new = u + x_new - z_new

        pr = np.linalg.norm(x_new - z_new) / (np.linalg.norm(x_new) + 1e-12)
        dr = rho * np.linalg.norm(z_new - z) / (np.linalg.norm(u_new) + 1e-12)
        residuals.append(float(pr))

        if (k+1) % adapt_freq == 0:
            if pr > tau * dr: rho *= mu
            elif dr > tau * pr: rho /= mu

        x, z, u = x_new, z_new, u_new
        if pr < tol:
            print(f"    Converged at iter {k+1} (res={pr:.2e})")
            break

    return np.clip(x, 0, None), residuals


if __name__ == "__main__":
    print("="*60); print("  FA-PnP-ADMM  (soft-thresholding)"); print("="*60)
    scene, raw, ref, params = get_shared_data()
    t0 = time.perf_counter()
    recon, res = fa_pnp_admm(raw, params)
    rt = time.perf_counter() - t0
    m  = compute_metrics(recon, ref)
    print(f"  Done {rt:.2f}s  PSNR={m['PSNR']:.2f}dB  SSIM={m['SSIM']:.4f}")
    save_comparison_figure(recon, ref, "FA-PnP-ADMM", m, rt, OUTPUT_DIR, res)
"""
algo4_RMA.py — Range Migration Algorithm (RMA / omega-k)
=========================================================
Classic frequency-domain SAR focusing algorithm.

In the convolution-model framework, RMA acts as a Wiener-like
deconvolution in the frequency domain:
  X_hat(f) = Y(f) * H*(f) / (|H(f)|^2 + epsilon)

where H(f) is the Fourier transform of the PSF.
This is the single-pass, non-iterative baseline.
"""

import numpy as np
from scipy.ndimage import convolve
import time, sys, os

sys.path.insert(0, os.path.dirname(__file__))
from sar_utils import get_shared_data, compute_metrics, save_comparison_figure, make_psf

OUTPUT_DIR = "sar_results"


def rma(raw, params):
    """
    Wiener deconvolution in 2-D frequency domain.

    Steps
    -----
    1. Compute PSF → zero-pad to image size → FFT
    2. FFT of observed image
    3. Wiener filter:  X = Y * conj(H) / (|H|^2 + eps)
    4. IFFT → focused image
    """
    Nr, Na = raw.shape
    psf    = make_psf(params, ksz=21)

    # Zero-pad PSF to image size (wrap PSF for circular convolution)
    psf_pad = np.zeros((Nr, Na))
    kh, kw  = psf.shape
    psf_pad[:kh, :kw] = psf
    # Shift so PSF centre is at [0,0]
    psf_pad = np.roll(psf_pad, -(kh//2), axis=0)
    psf_pad = np.roll(psf_pad, -(kw//2), axis=1)

    H   = np.fft.fft2(psf_pad)
    Y   = np.fft.fft2(raw.astype(np.float64))

    # Wiener regularisation — epsilon controls sharpness vs noise
    eps = 0.01 * np.mean(np.abs(H)**2)
    X   = Y * np.conj(H) / (np.abs(H)**2 + eps)

    focused = np.real(np.fft.ifft2(X))
    return np.clip(focused, 0, None)


if __name__ == "__main__":
    print("="*60); print("  RMA  (Wiener deconvolution / omega-k)"); print("="*60)
    scene, raw, ref, params = get_shared_data()
    t0 = time.perf_counter()
    recon = rma(raw, params)
    rt = time.perf_counter() - t0
    m  = compute_metrics(recon, ref)
    print(f"  Done {rt:.4f}s  (single pass)")
    print(f"  PSNR={m['PSNR']:.2f}dB  SSIM={m['SSIM']:.4f}")
    save_comparison_figure(recon, ref, "RMA", m, rt, OUTPUT_DIR, None)
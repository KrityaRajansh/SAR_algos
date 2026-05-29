# SAR Image Reconstruction — Algorithm Comparison Framework

A pure-NumPy simulation framework that benchmarks four SAR image reconstruction algorithms under identical experimental conditions, producing visual and quantitative comparisons suitable for research presentations.

---

## Algorithms Implemented

| File | Algorithm | Type | Denoiser |
|------|-----------|------|----------|
| `algo1_FA_PnP_ADMM.py` | **FA-PnP-ADMM** | Iterative | Soft-thresholding (wavelet shrinkage) |
| `algo2_FA_CNN_ADMM.py` | **FA-CNN-ADMM** | Iterative | 5-layer DnCNN-style convolutional network |
| `algo3_FA_AE_ADMM.py` | **FA-AE-ADMM** | Iterative | Residual patch autoencoder |
| `algo4_RMA.py` | **RMA** | Single-pass | Wiener deconvolution (ω-k algorithm) |

> **FA** = Fast & Adaptive · **PnP** = Plug-and-Play · **ADMM** = Alternating Direction Method of Multipliers · **RMA** = Range Migration Algorithm

---

## Forward Model

All algorithms solve the same inverse problem:

```
y = H * x + speckle + AWGN
```

where:
- `x` — clean SAR reflectivity scene (terrain, urban blocks, roads, forests, water, corner reflectors)
- `H` — 2D SAR Point Spread Function (sinc in range × Gaussian in azimuth)
- `speckle` — multiplicative Rayleigh fading noise (1-look)
- `AWGN` — additive white Gaussian noise at SNR = 15 dB

---

## Project Structure

```
SAR_algos/
│
├── sar_utils.py              # Shared simulation core (scene, PSF, forward model, metrics, plotting)
├── algo1_FA_PnP_ADMM.py      # FA-PnP-ADMM with soft-thresholding
├── algo2_FA_CNN_ADMM.py      # FA-CNN-ADMM with DnCNN-lite denoiser
├── algo3_FA_AE_ADMM.py       # FA-AE-ADMM with residual patch autoencoder
├── algo4_RMA.py              # Range Migration Algorithm (Wiener deconvolution)
├── run_all_experiments.py    # Master runner — all 4 experiments
├── requirements.txt          # Python dependencies
└── sar_results/              # Output folder (auto-created on run)
    ├── FA-PnP-ADMM_result.png
    ├── FA-CNN-ADMM_result.png
    ├── FA-AE-ADMM_result.png
    ├── RMA_result.png
    ├── SUMMARY_comparison.png
    ├── SUMMARY_metrics.png
    ├── E2_snr_sweep.png
    ├── E3_convergence.png
    ├── E4_radar_chart.png
    └── SUMMARY_table.csv
```

---

## Experiments

| # | Name | Description |
|---|------|-------------|
| E1 | Primary comparison | All 4 algorithms on the same scene at SNR = 15 dB |
| E2 | SNR sweep | PSNR and SSIM across SNR ∈ {5, 10, 15, 20, 25} dB |
| E3 | Convergence | Iteration vs residual plot for the three ADMM variants |
| E4 | Radar chart | Normalised spider chart across PSNR, SSIM, ENL, Contrast, Speed |

---

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **PSNR** (dB) | Peak Signal-to-Noise Ratio — higher is better |
| **SSIM** | Structural Similarity Index — higher is better (max 1.0) |
| **ENL** | Equivalent Number of Looks — speckle suppression quality |
| **Contrast** | Bright-target to background ratio |
| **Runtime** (s) | Wall-clock reconstruction time |

---

## Requirements

- Python ≥ 3.9
- NumPy
- SciPy
- scikit-image
- matplotlib

Install with:

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/SAR_algos.git
cd SAR_algos

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run everything
python run_all_experiments.py
```

Results are saved to `sar_results/`.

To run a single algorithm in isolation:

```bash
python algo1_FA_PnP_ADMM.py
python algo2_FA_CNN_ADMM.py
python algo3_FA_AE_ADMM.py
python algo4_RMA.py
```

---

## Key Design Choices

**Why pure NumPy / SciPy?**
No PyTorch or TensorFlow dependency. The CNN and AE denoisers use fixed random weights (deep-image-prior paradigm), making results fully reproducible without a GPU or pre-trained model files.

**Why the same scene for all algorithms?**
`sar_utils.get_shared_data()` uses a fixed random seed (`SEED = 42`), so every algorithm receives byte-identical input. This isolates reconstruction quality differences to the algorithm alone.

**ADMM x-update**
All three ADMM variants solve `(HᵀH + ρI)x = rhs` via 6-iteration conjugate gradient — exact enough for comparison purposes, fast enough to run on CPU.

**Adaptive ρ**
After every `adapt_freq` iterations, ρ is scaled up or down based on the primal/dual residual ratio following Boyd et al. (2011), giving the "Fast & Adaptive" property.

---

## Output Sample

Each algorithm produces a 6-panel figure:

```
[ Reference scene ] [ Reconstruction ] [ Error map      ]
[ Convergence     ] [ Metrics bars   ] [ Summary text   ]
```

Plus `SUMMARY_comparison.png` (all 4 side-by-side) and `SUMMARY_metrics.png` (grouped bar chart).

---

## References

- Boyd, S. et al. (2011). *Distributed Optimization and Statistical Learning via ADMM*. Foundations and Trends in Machine Learning.
- Venkatakrishnan, S. V. et al. (2013). *Plug-and-Play Priors for Model Based Reconstruction*. IEEE GlobalSIP.
- Zhang, K. et al. (2017). *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*. IEEE TIP.
- Ulyanov, D. et al. (2018). *Deep Image Prior*. CVPR.
- Cumming, I. G. & Wong, F. H. (2005). *Digital Processing of Synthetic Aperture Radar Data*. Artech House.

---

## License

MIT License — free to use, modify, and distribute with attribution.

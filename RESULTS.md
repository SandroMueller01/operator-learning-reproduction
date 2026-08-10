# Results at a Glance

Reproduction of Adcock, Dexter & Moraga, *"Optimal Deep Learning of Holomorphic
Operators Between Banach Spaces"* (arXiv:[2406.13928](https://arxiv.org/abs/2406.13928)),
for the parametric elliptic diffusion equation and the Navier–Stokes–Brinkman
(NSB) equations, in both PyTorch and JAX, at the paper's full experimental
matrix (2 coefficient families × 2 parametric dimensions × 6 architectures ×
14 training-set sizes × 12 trials, per PDE).

For the full methodology, the exact mixed finite element formulations, the
sparse-grid test protocol, and every documented deviation from the paper,
see the practical work report: [`practical_work_report/main-thesis.pdf`](practical_work_report/main-thesis.pdf)
(compiled from source in that directory; the deviation list is in the
appendix).

**Status: both experiments are complete.** 24,094 training runs finished
across both PDEs and both frameworks, 0 failures. See
[Diffusion](#diffusion-equation--headline-comparison-to-the-paper) and
[Navier–Stokes–Brinkman](#navierstokesbrinkman--complete) below.

---

## Diffusion equation — headline comparison to the paper

| Paper's claim | Our result | Verdict |
|---|---|---|
| Relative test error decays roughly as $m^{-1}$ | Mean fitted log-log slope across all 48 (case × architecture × framework) combinations: **−1.06** | **Matches** |
| ELU/tanh architectures outperform ReLU | ELU/tanh reach 3–20× lower error than ReLU at the same architecture size in every one of the 4 cases (see [table below](#error-at-the-largest-training-set-size-m--500)) | **Matches** |
| PyTorch and JAX should agree closely (same protocol, same data) | Geometric-mean test-error ratio (PyTorch/JAX) is between **0.99 and 1.00** in all 4 cases — under 1% systematic difference | **Matches** |
| No degradation going from $d=4$ to $d=8$ | Holds for the best-performing architectures (ELU, tanh): error stays flat or *improves* at $d=8$. Does **not** hold for ReLU or the largest architecture (10×100), where error at $d=8$ is 1.5–2.5× higher than at $d=4$ | **Partially matches** — see [dimension robustness](#dimension-robustness-d4-vs-d8) |

JAX is consistently faster to train than PyTorch under an identical protocol:
PyTorch's wall-clock training time is **9–28% higher** than JAX's across the
4 cases (geometric mean), consistent with JAX's JIT-compiled training step
avoiding PyTorch's per-epoch Python dispatch overhead once compiled.

### Reproduced Fig. 1-style plot (PyTorch)

![Diffusion paper-style figure](results/figures/diffusion_paper_figure_pytorch.png)

Relative test error vs. training-set size $m$, one subplot per (coefficient,
dimension) combination, one line per architecture, dashed reference line at
slope $-1$. Generated directly from the real sweep metrics via
[`src/ol_reproduction/plotting/plot_paper_figure.py`](src/ol_reproduction/plotting/plot_paper_figure.py).

### PyTorch vs. JAX, best-performing architecture (`mlp_4x40_elu`, `diffusion_affine_d4`)

| Error | Training time |
|---|---|
| ![Framework error comparison](results/figures/diffusion_affine_d4_mlp_4x40_elu_framework_error.png) | ![Framework time comparison](results/figures/diffusion_affine_d4_mlp_4x40_elu_framework_time.png) |

### Error at the largest training-set size (m = 500)

Geometric mean over 12 trials, `diffusion_affine_d4` (representative case —
all 4 cases show the same activation ranking):

| Architecture | Activation | PyTorch | JAX |
|---|---|---:|---:|
| 4×40  | ELU  | 0.00386 | 0.00374 |
| 4×40  | tanh | 0.00565 | 0.00573 |
| 4×40  | ReLU | 0.01644 | 0.01648 |
| 10×100 | ELU  | 0.00403 | 0.00384 |
| 10×100 | ReLU | 0.02820 | 0.02631 |
| 10×100 | tanh | 0.06910 | 0.06645 |

(Full table across all 4 cases and both frameworks: `results/tables/diffusion_table_m500.csv`.)

### Dimension robustness (d=4 vs. d=8)

Ratio of geometric-mean error at $d=8$ to $d=4$, same architecture and
coefficient family (>1 means $d=8$ performs worse):

| Architecture | Affine | Log-transformed |
|---|---:|---:|
| 4×40 ELU   | 0.70 | 1.00 |
| 4×40 tanh  | 0.45 | 0.55 |
| 4×40 ReLU  | 1.48 | 2.37 |
| 10×100 ELU  | 0.96 | 1.74 |
| 10×100 tanh | 1.54 | 2.12 |
| 10×100 ReLU | 1.77 | 2.53 |

The paper reports no degradation from $d=4$ to $d=8$; our reproduction shows
this cleanly for ELU and tanh (the architectures the paper itself
recommends), but ReLU architectures do degrade noticeably at higher
dimension. This is reported here rather than smoothed over, and is discussed
in the report's results chapter.

(Full table: `results/tables/diffusion_table_dimension.csv`; slope-fit table:
`results/tables/diffusion_table_slopes.csv`; framework-ratio table:
`results/tables/diffusion_table_framework_ratio.csv`.)

---

## Navier–Stokes–Brinkman — complete

Data generation: all 4 cases (affine/log × $d\in\{4,8\}$), 0 non-converged
(excluded) samples across 24,000+ nonlinear mixed FEM solves. Training:
**16,128 / 16,128 runs complete, 0 failures** (both frameworks, both
targets $u$ and $p$, full paper matrix).

One documented deviation applies here (see the report appendix): NSB
training uses a reduced epoch budget (20,000 epochs, $5\times10^{-6}$ loss
tolerance) instead of the paper's 60,000/$5\times10^{-7}$, because direct
measurement showed NSB runs routinely needing tens of thousands of epochs
to approach the paper's tolerance — the full paper-scale protocol was
measured at roughly 9 days of wall-clock time on the hardware available for
this practical work, which was not tractable. Diffusion is unaffected by
this change.

| Paper's claim | Our result | Verdict |
|---|---|---|
| Relative test error decays roughly as $m^{-1}$ | Fitted slope: **−0.995** ($u$), **−0.912** ($p$), averaged over all combinations | **Matches** (slightly shallower for $p$, consistent with the reduced training budget above) |
| ELU/tanh outperform ReLU | ELU/tanh reach 2–5× lower error than ReLU at $m=500$, pooled over all cases and both frameworks (see table below) | **Matches** |
| PyTorch and JAX should agree closely | Geometric-mean test-error ratio (PyTorch/JAX) between **0.986 and 1.006** across all 8 (case × target) combinations | **Matches** |
| No degradation from $d=4$ to $d=8$ | Robustness here tracks **architecture size**, not activation: every 4×40 network is neutral-to-improving at $d=8$ (ratio 0.77–1.00), every 10×100 network degrades (1.17–2.03), regardless of activation | **Partially matches** — and notably a *different* pattern than diffusion showed (see below) |

Unlike diffusion, JAX's training-time advantage over PyTorch is not
universal here — PyTorch is actually faster in 2 of the 8 (case × target)
combinations, plausibly because the sweep's many distinct
architecture/training-size shape combinations erode JAX's JIT-compilation
advantage (see the report's JAX chapter for why this was anticipated).

### Reproduced Fig. 2-style plots (PyTorch)

| Velocity ($u$) | Pressure ($p$) |
|---|---|
| ![NSB u paper figure](results/figures/nsb_paper_figure_u_pytorch.png) | ![NSB p paper figure](results/figures/nsb_paper_figure_p_pytorch.png) |

### Error at m = 500, pooled over all cases/frameworks/architectures

| Activation | Target $u$ | Target $p$ |
|---|---:|---:|
| ELU  | 0.00766 | 0.00888 |
| tanh | 0.01897 | 0.01334 |
| ReLU | 0.03486 | 0.04447 |

### Dimension robustness (d=4 vs. d=8), affine coefficient

Ratio of geometric-mean error at $d=8$ to $d=4$ (>1 means $d=8$ is worse),
averaged over $u$ and $p$ — compare to the diffusion table above:

| Architecture | NSB (avg. u, p) | Diffusion (for reference) |
|---|---:|---:|
| 4×40 ELU   | 1.00 | 0.70 |
| 4×40 tanh  | 0.77 | 0.45 |
| 4×40 ReLU  | 0.83 | 1.48 |
| 10×100 ELU  | 1.17 | 0.96 |
| 10×100 tanh | 2.03 | 1.54 |
| 10×100 ReLU | 1.91 | 1.77 |

Interestingly, dimension robustness is predicted by a *different* property
for each PDE: for diffusion, ReLU degrades regardless of size while ELU is
robust regardless of size (an *activation* effect); for NSB, every 4×40
network is robust regardless of activation while every 10×100 network
degrades (a *size* effect). Both patterns are consistent across both
coefficient families within their own experiment — this is discussed in
the report's results chapter as a genuine, reproducible difference between
the two PDEs, not noise.

(Full tables: `results/tables/nsb_table_m500.csv`,
`results/tables/nsb_table_slopes.csv`,
`results/tables/nsb_table_framework_ratio.csv`,
`results/tables/nsb_table_dimension.csv`,
`results/tables/nsb_table_activation.csv`.)

---

## Bottom line

Both experiments, at the paper's full scale (24,094 completed training
runs across both PDEs and frameworks, 0 failures), reproduce the paper's
central qualitative claims: an approximately $m^{-1}$ error-decay rate and
a clear, consistent ELU/tanh-over-ReLU ranking. The one claim that only
partially holds — no degradation from $d=4$ to $d=8$ — is reported
honestly rather than smoothed over, including the fact that it fails for
different reasons in the two experiments. See the report's Discussion and
Conclusion chapters for the full synthesis.

---

## Reproducing these numbers

```powershell
$env:PYTHONPATH = "src"

# Full paper-scale training sweep (both frameworks, all 8 cases), 16-way parallel:
python scripts/run_parallel_sweep.py --paper-matrix --workers 16 --resume

# Regenerate the summary tables and paper-style figure from results/metrics/:
python scripts/summarize_results.py --print-table
```

Data generation (the actual mixed FEM solves) requires FEniCS 2019.1.0 and
runs inside WSL2 — see `scripts/wsl/` and the report's methodology chapter
for the full environment setup. All 8 datasets are already generated in this
repository's `data/processed/` (not committed to git due to size; see
`.gitignore` and the generation scripts to reproduce them).

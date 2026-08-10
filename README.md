# Operator Learning Reproduction

A practical-work reproduction of Adcock, Dexter & Moraga,
**"Optimal Deep Learning of Holomorphic Operators Between Banach Spaces"**
(arXiv:[2406.13928](https://arxiv.org/abs/2406.13928)).

**→ For a quick look at how well the reproduction matches the paper, see [`RESULTS.md`](RESULTS.md).**

**→ For the full write-up (methodology, theory summary, implementation details,
and every documented deviation from the paper), see the compiled report:
[`practical_work_report/main-thesis.pdf`](practical_work_report/main-thesis.pdf).**

## What this reproduces

Two of the paper's three numerical experiments, at the paper's stated
experimental scale, using the paper's actual data-generation and evaluation
protocol rather than a simplified stand-in:

| Experiment | Output space | PDE discretization | Status |
|---|---|---|---|
| Parametric elliptic diffusion | Hilbert-valued ($L^2$) | Mixed FEM, $RT_1 \times DG_0$ | Data + training complete |
| Navier–Stokes–Brinkman (NSB) | Banach-valued ($\mathbf{L}^4$) | Mixed FEM, Arnold–Falk–Winther-type elements | Data complete, training in progress |

The stationary Boussinesq equation (the paper's third experiment) is out of
scope for this practical work.

Both experiments are trained with fully connected networks in **both
PyTorch and JAX**, across the paper's full matrix: 2 coefficient families
(affine, log-transformed) × 2 parametric dimensions ($d \in \{4, 8\}$) × 6
architectures (4×40 / 10×100 hidden layers, ReLU / ELU / tanh) × 14
training-set sizes × 12 independent trials.

Every point where this reproduction necessarily deviates from an
underspecified or ambiguous detail in the paper (e.g. the exact NSB finite
element family, which the paper doesn't state explicitly) is documented in
the report's appendix, not left implicit.

## Repository structure

```text
operator-learning-reproduction/
├── RESULTS.md                    <- start here for a quick results overview
├── practical_work_report/        <- the full LaTeX report (compile: xelatex -> biber -> xelatex x2)
│
├── configs/
│   ├── pde/                      <- 8 materialized (coefficient x dimension) configs
│   ├── model/                    <- 6 architecture configs
│   └── train/                    <- PyTorch/JAX training configs (paper-spec + NSB's reduced-budget variant)
│
├── data/processed/                <- generated datasets (not committed; see Data Generation below)
├── results/
│   ├── metrics/                   <- one CSV per (case, target, framework, architecture)
│   ├── figures/                   <- generated plots
│   ├── tables/                    <- generated summary tables
│   └── logs/                      <- sweep run logs
│
├── scripts/
│   ├── generate_data.py           <- single-case data generation (finite-difference fallback path)
│   ├── wsl/                       <- WSL2/FEniCS mixed-FEM data generation (the real pipeline)
│   ├── materialize_pde_configs.py <- expands the 2 grouped PDE configs into 8 flat ones
│   ├── train_pytorch.py / train_jax.py       <- single training run
│   ├── run_pytorch_sweep.py / run_jax_sweep.py <- sweep over one (case, target, architecture)
│   ├── run_parallel_sweep.py      <- process-pool-parallel full-matrix sweep driver (use this)
│   ├── run_all_available.py       <- serial orchestrator (tests -> data -> sweeps -> plots)
│   ├── plot_error_vs_m.py / plot_framework_comparison.py
│   └── summarize_results.py
│
├── src/ol_reproduction/
│   ├── coefficients/               <- affine.py, log_transformed.py (paper eq. B.2)
│   ├── config/                     <- YAML loading / config resolution
│   ├── data/                       <- sampling, sparse_grid.py (Clenshaw-Curtis via Tasmanian), dataset I/O
│   ├── evaluation/                 <- relative_error.py (mass-matrix-weighted), framework_comparision.py
│   ├── models/                     <- pytorch_mlp.py, jax_mlp.py
│   ├── pde/
│   │   ├── diffusion/fenics_mixed_solver.py         <- real mixed FEM diffusion solver
│   │   ├── navier_stokes_brinkman/fenics_solver.py  <- real nonlinear mixed FEM NSB solver
│   │   └── mass_matrix.py
│   ├── plotting/                   <- plot_paper_figure.py (paper Fig. 1/2-style), aggregation.py (geometric mean/std)
│   └── training/                   <- pytorch_train.py, jax_train.py (checkpoint/restore rule, LR schedule)
│
└── tests/
```

## Data generation

The real pipeline solves the paper's mixed finite element formulations with
FEniCS 2019.1.0, which requires WSL2 on Windows (legacy FEniCS is not
installable natively). See `scripts/wsl/` for the setup scripts and case
generators:

```bash
# inside WSL2, fenics2019 conda env, from the repo root
python scripts/wsl/generate_diffusion_case.py --case-name diffusion_affine_d4 --coefficient affine --dimension 4
python scripts/wsl/generate_nsb_case.py --case-name nsb_affine_d4 --coefficient affine --dimension 4 --workers 28
```

Each case produces, under `data/processed/<case_name>/`:

```text
train_m{m}_seed{s}.npz   <- one file per (training-set size, trial), arrays: x, y_u [, y_p]
test.npz                  <- shared deterministic Clenshaw-Curtis sparse-grid test set: x, w, y_u [, y_p]
mass_matrix[_u/_p].npz    <- FEM mass matrix, for the Bochner-norm test error
metadata.json              <- mesh/DOF/sparse-grid/Newton-convergence provenance
```

`generate_nsb_case.py --workers N` solves the (embarrassingly parallel)
per-sample nonlinear systems across a process pool — this was ~89× faster in
practice than the original serial version, and is safely resumable if
interrupted (it skips any trial whose output files already exist).

The finite-difference solvers under `scripts/generate_data.py` /
`src/ol_reproduction/pde/*/fd_solver.py` predate the mixed-FEM pipeline and
remain only as a quick, FEniCS-free smoke-test path for the training and
sweep code — they are **not** used for any of the reported results.

## Training

Training reads the pre-generated `.npz` files directly and never needs
FEniCS. A single run:

```powershell
$env:PYTHONPATH = "src"

python scripts/train_pytorch.py `
  --dataset data/processed/diffusion_affine_d4 `
  --train-file train_m500_seed0.npz `
  --test-file test.npz `
  --target u `
  --model configs/model/mlp_4x40_elu.yaml `
  --train configs/train/pytorch_paper.yaml
```

The full paper-matrix sweep, run across a 16-way process pool with
automatic resume:

```powershell
$env:PYTHONPATH = "src"

python scripts/run_parallel_sweep.py --paper-matrix --workers 16 --resume
```

Each worker trains one (case, target, architecture, framework, $m$, trial)
combination independently and appends one row to
`results/metrics/<case>_<target>_<framework>_<architecture>.csv`. NSB uses a
documented reduced training budget (`configs/train/pytorch_nsb.yaml` /
`jax_nsb.yaml`) instead of the paper's full 60,000-epoch protocol — see the
report appendix for why.

## Evaluation, plots, and tables

```powershell
$env:PYTHONPATH = "src"

python scripts/summarize_results.py --print-table
python scripts/plot_framework_comparison.py `
  --pytorch-metrics results/metrics/diffusion_affine_d4_u_pytorch_mlp_4x40_elu.csv `
  --jax-metrics results/metrics/diffusion_affine_d4_u_jax_mlp_4x40_elu.csv `
  --output-dir results/figures `
  --experiment-name diffusion_affine_d4_mlp_4x40_elu
```

Test error is measured in the FEM-mass-matrix-weighted (Bochner) norm, not a
naive Euclidean sum-of-squares, and trial-level aggregation uses geometric
mean/std (matching the paper's own log-log-axis convention), implemented in
`src/ol_reproduction/plotting/aggregation.py`.

## Testing

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests
```

FEniCS- and Tasmanian-dependent tests are collected and run separately
inside the corresponding WSL2 conda environments; the Windows-side suite
covers everything else (config loading, coefficients, models, training
loops, evaluation, plotting, sweep resumption).

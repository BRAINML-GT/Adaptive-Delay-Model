# Adaptive Delay Model (ADM)

**Official PyTorch implementation** of

> **Learning Time-Varying Multi-Region Brain Communications via Scalable Markovian Gaussian Processes**
> Weihan Li, Yule Wang, Chengrui Li, Anqi Wu
> *Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025 (Oral)*
> [arXiv:2407.00397](https://arxiv.org/abs/2407.00397)
> · [PMLR](https://proceedings.mlr.press/v267/li25ck.html)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

ADM models multi-region neural recordings via a shared GP latent with
**time-varying inter-region communication delays** `δ(t)`. The GP prior
is lifted into a Markov state-space model and inferred with a parallel-scan
Kalman EM (O(log T) work-depth on GPU).

This repo ships the **ADM paper artifacts**:

- `examples/demo_synthetic.py` — multi-region delay-recovery on
  synthetic data (ground-truth δ(t) → recovered δ(t))
- `examples/demo_v1v2.py` — V1/V2 visual-cortex recordings (NLB-style
  held-out-neuron co-smoothing)
- `notebooks/` — Jupyter notebooks with results baked in

The ADM model and training infrastructure live in the
**[`mbrila`](https://github.com/BRAINML-GT/MBRILA) GP-SSM framework**
— this repo only contains the paper-specific demos, data, and README.

---

## Installation

Install the [`mbrila`](https://github.com/BRAINML-GT/MBRILA) framework
first (tested with `v0.1.0`):

```bash
git clone https://github.com/BRAINML-GT/MBRILA.git mbrila
cd mbrila
git checkout v0.1.0
uv sync              # or:  pip install -e .
cd ..
```

Then clone this repo and run the demos inside that environment:

```bash
git clone https://github.com/BRAINML-GT/Adaptive-Delay-Model.git
cd Adaptive-Delay-Model
source ../mbrila/.venv/bin/activate   # share mbrila's environment
```

> If you encounter API errors with a newer `mbrila`, check out the tag
> `v0.1.0` — that is the version this repo is tested against.

---

## Quickstart

**Synthetic delay-recovery** — 5-region scenario, K=2 across-latents,
Gaussian time-varying delays, heterogeneous per-latent timescales:

```bash
python examples/demo_synthetic.py --out-dir demo_outputs/synthetic
```

Outputs: convergence trace, per-pair `δ_recovered(t)` overlaid on truth,
per-region smoother latents, `y` reconstruction, `summary.json`.

**Real V1/V2 visual cortex** — 2-region, 400 trials, co-smoothing
on held-out neurons:

```bash
python examples/demo_v1v2.py --out-dir demo_outputs/v1v2
```

Outputs: convergence trace (train + val), fitted δ(t), smoother latents,
PSTH and trial-0 reconstruction heatmaps, `eval_metrics.txt`.

For interactive exploration with figures inline:

```bash
jupyter lab notebooks/demo_synthetic.ipynb
jupyter lab notebooks/demo_v1v2.ipynb
```

---

## Data

`data/demo_v1v2_data.pkl` is one recording session from the **Semedo et
al., Neuron 2019** V1/V2 dataset (400 trials × 64 bins × 72 V1 + 22 V2
neurons), with spike counts Gaussian-smoothed and z-scored to match
the linear-Gaussian emission model.

If you use the V1/V2 data, cite:

> Semedo, J. D., Zandvakili, A., Machens, C. K., Yu, B. M., & Kohn, A.
> *Cortical Areas Interact through a Communication Subspace*. Neuron,
> 102(1), 249-259.e4 (2019). https://doi.org/10.1016/j.neuron.2019.01.026

---

## Citation

If you use ADM, please cite:

```bibtex
@inproceedings{li2025learning,
  title={Learning Time-Varying Multi-Region Brain Communications via Scalable Markovian Gaussian Processes},
  author={Li, Weihan and Wang, Yule and Li, Chengrui and Wu, Anqi},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  pages={36021--36041},
  year={2025},
  organization={PMLR}
}
```

---

## See also

- **[`mbrila`](https://github.com/BRAINML-GT/MBRILA)** — the underlying
  GP-SSM framework. It also ships other multi-region presets (`DLAG` /
  `mDLAG` / `GPFA-SSM` / `LDS`) that share the same kernel / engine /
  observation infrastructure, making them directly comparable as
  baselines.

## License

MIT — see [LICENSE](LICENSE).

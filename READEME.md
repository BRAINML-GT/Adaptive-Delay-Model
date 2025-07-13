# Adaptive Delay Model (ADM) 

**Title:** Learning Time-Varying Multi-Region Communications via Scalable Markovian Gaussian Processes (ICML 2025 Oral)   
**Authors:** Weihan Li, Yule Wang, Chengrui Li, Anqi Wu  
**Paper:** [ArXiv:2407.00397](https://arxiv.org/pdf/2407.00397)

The Python implementation of the **Adaptive Delay Model (ADM)** for learning time-varying multi-region neural communications via Markovian Gaussian Processes (State Space Model).

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  

---

## 🌟 Features

- **Time-varying delay estimation**  
  Learns continuous, time-varying temporal delays between brain regions.

- **Markovian GP ↔ SSM conversion**  
  Bridges arbitrary stationary temporal Gaussian Processes and State Space Models.

- **Parallel-scan Kalman EM**  
  O(log T) inference using parallel scan algorithms.

- **Scalable to large datasets**  
  Handles long neural recordings and multiple brain regions efficiently.

---

## 🚀 Installation

```bash
# Clone the repo
git clone https://github.com/BRAINML-GT/Adaptive-Delay-Model.git
cd Adaptive-Delay-Model

# (Recommended) Create a conda environment
conda create -n adm-env python=3.13
conda activate adm-env

# Install dependencies
pip install -r requirements.txt
```


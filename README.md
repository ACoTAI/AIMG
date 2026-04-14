# AIMG

This repository provides the official implementation of the paper:

> **AIMG: Adaptive Incentive Model on Graph for Cooperative Multi-Agent Systems**

---

## 🔥 Introduction

AIMG is a novel framework for cooperative multi-agent reinforcement learning (MARL), which introduces an **adaptive incentive mechanism on graph structures** to enhance coordination and cooperation among agents under partial observability.

Specifically, AIMG models agent interactions through dynamic graph structures and designs an incentive-driven learning paradigm to mitigate inefficient or unproductive behaviors in cooperative tasks.

---

## 📦 Codebase

This project is built upon the following open-source frameworks:

- **PyMARL2** (baseline MARL framework):  
  https://github.com/hijkzzz/pymarl2

- **SMAC (StarCraft Multi-Agent Challenge)**:  
  https://github.com/andrewmarx/smac

- **Google Research Football (GRF)**:  
  https://github.com/google-research/football

We sincerely thank the authors of these repositories for their valuable contributions.

---

## ⚙ Installation

### 1. Create Environment

We recommend using Conda:

```bash
# Require Anaconda3 or Miniconda3
conda create -n pymarl python=3.8 -y
conda activate pymarl
```


## Install StarCraft II and SMAC


```bash
bash install_dependencies.sh
```
Make sure StarCraft II version 2.4.10 is correctly installed for SMAC.

## Install Google Research Football (Optional)

```bash
bash install_gfootball.sh
```

## Running Experiments

```bash
python src/main.py --config=xxx --env-config=xxx with env_args.map_name=xxx
```
## Citation
If you find this code useful, please cite:
```bash
@article{Zhou2026AIMG,
  title={AIMG: Adaptive Incentive Model on Graph for Cooperative Multi-Agent Systems},
  author={Changdong Zhou, Yubo Ma, Ruofan Hu, Bo Zhang, Jian Wan, C. L. Philip Chen, Hongbo Liu},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2026}
}
```

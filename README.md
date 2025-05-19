# HalfCheetah-v4 Imitation Learning

This project implements and compares different imitation learning approaches for the HalfCheetah-v4 environment.

## Project Organization

```
halfcheetah_v4_sac/
├── actor_impl.py
├── data/
├── models/
├── results/
├── plots/
├── train_logprob.py
├── train_mog.py
├── train_diffusion.py
├── train_autoreg_disc.py
├── train.py
├── compare_methods.py
└── evaluate.py
```

## Imitation Learning Methods

- Log Probability Minimization
- Mixture of Gaussians
- Diffusion Policy
- Autoregressive Discretized
# HalfCheetah-v4 Imitation Learning

This project implements and compares different imitation learning approaches for the HalfCheetah-v4 environment using PyTorch.

## Project Organization

```
halfcheetah_v4_sac/
├── actor_impl.py           # Base actor implementation (do not modify)
├── data/                   # Original data and weights
│   ├── halfcheetah_v4_actor_weights.pt   # Expert policy weights
│   └── halfcheetah_v4_data.pt            # Expert demonstrations data
├── models/                 # Trained model weights
│   ├── logprob_*.pt        # Log probability models
│   └── mog_*.pt            # Mixture of Gaussians models
├── results/                # Training and evaluation results
│   ├── logprob/            # Log probability method results
│   ├── mog/                # Mixture of Gaussians results
│   └── evaluations/        # Evaluation results
├── plots/                  # Generated plots
├── train_logprob.py        # Script for log probability minimization
├── train_mog.py            # Script for Mixture of Gaussians approach
└── evaluate.py             # General evaluation script
```

## Imitation Learning Methods

The project currently implements:

1. **Negative Log Probability Minimization** (`train_logprob.py`): 
   - The simplest approach to imitation learning
   - Directly optimizes a policy to maximize the likelihood of expert actions

2. **Mixture of Gaussians** (`train_mog.py`):
   - Models the action distribution as a weighted mixture of Gaussian components
   - Better captures multimodal action distributions
   - Configured with 5 components by default

## Training

To train a model using negative log probability minimization:

```bash
python train_logprob.py
```

To train a model using Mixture of Gaussians:

```bash
python train_mog.py
```

Training results are automatically saved to:
- Model weights in `models/`
- Training metrics in `results/`
- Learning curves in `plots/`

## Evaluation

To evaluate a trained model:

```bash
python evaluate.py --model models/mog_5_best.pt --episodes 20
```

Options:
- `--model`: Path to the model to evaluate (default: models/logprob_best.pt)
- `--episodes`: Number of episodes to evaluate (default: 20)
- `--render`: Enable rendering of the environment
- `--no-save`: Don't save evaluation results

The evaluation results are saved to `results/evaluations/`.

## References

### Negative Log Probability Minimization
- Behavioral Cloning (BC) - A staple technique in imitation learning
- Based on supervised learning principles

### Mixture of Gaussians
- Bishop, C. M. (1994). "Mixture Density Networks"
- Codevilla, F. et al. (2018). "End-to-end Driving via Conditional Imitation Learning"

## Report Generation

The saved JSON files can be used to generate LaTeX reports comparing different methods. Results are structured in a consistent format to make comparisons easy. 
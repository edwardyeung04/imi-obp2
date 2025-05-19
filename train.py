import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import json
import time
import argparse
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader, random_split
from actor_impl import Actor

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data():
    print("Loading data...")
    data = torch.load('data/halfcheetah_v4_data.pt', map_location=device, weights_only=False)
    observations = data['observations'].to(device)
    actions = data['actions'].to(device)
    print(f"Loaded {len(observations)} observation-action pairs")
    print(f"Mean reward in dataset: {data['mean_reward']}, Std: {data['std_reward']}")
    return observations, actions

def setup_env():
    env = gym.make("HalfCheetah-v4")
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    return env

def evaluate_model(model, env, num_episodes=10, **kwargs):
    rewards = []
    model_type = kwargs.get('model_type', 'actor')
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                
                if model_type == 'diffusion':
                    action = model.sample(obs_tensor)
                elif model_type == 'mog':
                    use_mixture = kwargs.get('use_mixture', False)
                    action, _, _ = model.get_action(obs_tensor, use_mixture=use_mixture)
                elif model_type == 'autoreg':
                    action = model.get_action(obs_tensor, greedy=kwargs.get('greedy', False))
                elif model_type == 'logprob':
                    action = model(obs_tensor)
                else:
                    _, _, action = model.get_action(obs_tensor)
                
                action = action.cpu().squeeze().numpy()
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)

def create_datasets(observations, actions, batch_size, train_ratio=0.9):
    dataset = TensorDataset(observations, actions)
    
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)
    
    return train_loader, val_loader

def save_results(method, final_reward, training_time, losses, val_losses, rewards, eval_epochs, hyperparams=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f'results/{method}', exist_ok=True)
    
    safe_hyperparams = {}
    if hyperparams:
        for key, value in hyperparams.items():
            if key not in ['env', 'device']:
                if hasattr(value, 'item'):
                    safe_hyperparams[key] = value.item()
                else:
                    safe_hyperparams[key] = value
    
    results = {
        "method": method,
        "timestamp": timestamp,
        "final_reward": float(final_reward),
        "training_time_seconds": training_time,
        "loss_history": [float(x) for x in losses],
        "val_loss_history": [float(x) for x in val_losses] if val_losses is not None else [],
        "reward_history": [float(x) for x in rewards],
        "eval_epochs": eval_epochs,
        "hyperparameters": safe_hyperparams
    }
    
    filename = f"results/{method}/{method}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename

def plot_training_curves(method, losses, val_losses, rewards, eval_epochs):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(losses)+1), losses, label='Train Loss')
    if val_losses is not None and len(val_losses) > 0:
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.title(f'{method} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(eval_epochs, rewards, marker='o')
    plt.title(f'{method} Evaluation Rewards')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/{method}_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training curves saved to {plot_path}")
    return plot_path

def main():
    description = "Train imitation learning models for HalfCheetah-v4 environment"
    epilog = '''
Examples:
  python train.py --method logprob                 # Train log probability model with default settings
  python train.py --method mog --batch_size 2048   # Train mixture of gaussians with batch size 2048
  python train.py --method diffusion --epochs 30   # Train diffusion policy for 30 epochs
  python train.py --method autoreg --lr 1e-4       # Train autoregressive model with learning rate 1e-4
  
Method descriptions:
  logprob:   Simple behavior cloning with MSE loss
  mog:       Mixture of Gaussians model (best performing)
  diffusion: Diffusion-based policy
  autoreg:   Autoregressive discretized model with categorical distribution
    '''
    
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--method', type=str, required=True, 
                      choices=['logprob', 'mog', 'diffusion', 'autoreg'],
                      help='Which training method to use (required)')
    parser.add_argument('--batch_size', type=int, default=4096,
                      help='Batch size for training (default: 4096)')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Learning rate (default: 3e-4)')
    parser.add_argument('--eval_interval', type=int, default=5,
                      help='Evaluate model every N epochs (default: 5)')
    
    parser.add_argument('--num_components', type=int, default=5,
                      help='Number of mixture components for MoG method (default: 5)')
    parser.add_argument('--num_bins', type=int, default=21,
                      help='Number of bins for discretization in autoreg method (default: 21)')
    parser.add_argument('--n_timesteps', type=int, default=100,
                      help='Number of diffusion timesteps for diffusion method (default: 100)')
    parser.add_argument('--use_mixture', action='store_true',
                      help='Use mixture sampling for MoG method during evaluation')
    
    args = parser.parse_args()
    
    if args.method == 'logprob':
        from train_logprob import LogProbModel, train_logprob
        model_type = 'logprob'
        ModelClass = LogProbModel
        train_fn = train_logprob
    elif args.method == 'mog':
        from train_mog import MoGActor, train_mog
        model_type = 'mog'
        ModelClass = MoGActor
        train_fn = train_mog
    elif args.method == 'diffusion':
        from train_diffusion import DiffusionPolicy, train_diffusion
        model_type = 'diffusion'
        ModelClass = DiffusionPolicy
        train_fn = train_diffusion
    elif args.method == 'autoreg':
        from train_autoreg_disc import AutoregressiveDiscretizedModel, train_autoreg_disc
        model_type = 'autoreg'
        ModelClass = AutoregressiveDiscretizedModel
        train_fn = train_autoreg_disc
    
    env = setup_env()
    observations, actions = load_data()
    model = ModelClass(env).to(device)
    
    train_args = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'eval_interval': args.eval_interval,
        'env': env,
        'device': device,
        'model_type': model_type,
    }
    
    if args.method == 'mog':
        train_args['num_components'] = args.num_components
        train_args['use_mixture'] = args.use_mixture
    elif args.method == 'autoreg':
        train_args['num_bins'] = args.num_bins
    elif args.method == 'diffusion':
        train_args['n_timesteps'] = args.n_timesteps
    
    print(f"\n===== Training with {args.method} method =====")
    results = train_fn(model, observations, actions, **train_args)
    
    save_results(
        args.method,
        results.get('final_reward', results.get('best_reward', 0)),
        results.get('training_time', 0),
        results.get('losses', []),
        results.get('val_losses', []),
        results.get('rewards', []),
        results.get('eval_epochs', []),
        train_args
    )
    
    plot_training_curves(
        args.method,
        results.get('losses', []),
        results.get('val_losses', []),
        results.get('rewards', []),
        results.get('eval_epochs', [])
    )
    
    print(f"\nTraining completed! Best reward: {results.get('best_reward', 'N/A')}")

if __name__ == "__main__":
    main() 
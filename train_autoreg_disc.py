import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from actor_impl import Actor
import time
import os
import json
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader, random_split
from train import create_datasets, evaluate_model, device

os.makedirs('models', exist_ok=True)
os.makedirs('results/autoreg_disc', exist_ok=True)
os.makedirs('plots', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data...")
data = torch.load('data/halfcheetah_v4_data.pt', map_location=device, weights_only=False)
observations = data['observations'].to(device)
actions = data['actions'].to(device)
print(f"Loaded {len(observations)} observation-action pairs")
print(f"Mean reward in dataset: {data['mean_reward']}, Std: {data['std_reward']}")

env = gym.make("HalfCheetah-v4")
env.single_observation_space = env.observation_space
env.single_action_space = env.action_space

NUM_BINS = 21
ACTION_DIM = np.prod(env.single_action_space.shape).item()

class AutoregressiveDiscretizedModel(nn.Module):
    def __init__(self, env, num_bins=NUM_BINS, hidden_dim=256):
        super().__init__()
        self.obs_dim = np.array(env.single_observation_space.shape).prod()
        self.action_dim = np.prod(env.single_action_space.shape).item()
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim * num_bins)
        )
        
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        
        bin_edges = torch.linspace(-1, 1, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.register_buffer("bin_centers", bin_centers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def discretize_actions(self, actions):
        normalized_actions = (actions - self.action_bias) / self.action_scale
        normalized_actions = torch.clamp(normalized_actions, -1, 1)
        
        indices = ((normalized_actions + 1) / 2 * (self.num_bins - 1)).long()
        indices = torch.clamp(indices, 0, self.num_bins - 1)
        
        return indices
    
    def continuous_from_discrete(self, discrete_actions):
        continuous_normalized = self.bin_centers[discrete_actions]
        continuous_actions = continuous_normalized * self.action_scale + self.action_bias
        return continuous_actions
    
    def forward(self, obs):
        batch_size = obs.shape[0]
        base_features = self.shared_encoder(obs)
        prev_actions_placeholder = torch.zeros(
            batch_size, self.action_dim, device=obs.device
        )
        combined = torch.cat([base_features, prev_actions_placeholder], dim=1)
        all_logits = self.action_predictor(combined)
        all_logits = all_logits.view(batch_size, self.action_dim, self.num_bins)
        return all_logits
    
    def get_action(self, obs, greedy=False):
        batch_size = obs.shape[0]
        self.eval()
        
        with torch.no_grad():
            base_features = self.shared_encoder(obs)
            discrete_actions = torch.zeros(
                batch_size, self.action_dim, dtype=torch.long, device=obs.device
            )
            prev_actions = torch.zeros(
                batch_size, self.action_dim, device=obs.device
            )
            
            for i in range(self.action_dim):
                combined = torch.cat([base_features, prev_actions], dim=1)
                all_logits = self.action_predictor(combined)
                all_logits = all_logits.view(batch_size, self.action_dim, self.num_bins)
                current_logits = all_logits[:, i, :]
                
                if greedy:
                    current_action = torch.argmax(current_logits, dim=1)
                else:
                    probs = F.softmax(current_logits, dim=1)
                    current_action = torch.multinomial(probs, 1).squeeze(-1)
                
                discrete_actions[:, i] = current_action
                
                if i < self.action_dim - 1:
                    prev_actions[:, i] = self.bin_centers[current_action]
            
            continuous_actions = self.continuous_from_discrete(discrete_actions)
            return continuous_actions

def train_autoreg_disc(model, observations, actions, **kwargs):
    batch_size = kwargs.get('batch_size', 2048)
    epochs = kwargs.get('epochs', 50)
    lr = kwargs.get('lr', 1e-4)
    eval_interval = kwargs.get('eval_interval', 5)
    env = kwargs.get('env', None)
    num_bins = kwargs.get('num_bins', NUM_BINS)
    
    if env is None:
        from train import setup_env
        env = setup_env()
    
    train_loader, val_loader = create_datasets(observations, actions, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    val_losses = []
    eval_rewards = []
    eval_rewards_greedy = []
    eval_epochs = []
    best_reward = -float('inf')
    best_model_state = None
    
    print(f"Training autoregressive discretized model for {epochs} epochs with batch size {batch_size}")
    print(f"Steps per epoch: {len(train_loader)}")
    
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for obs_batch, actions_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            discrete_actions = model.discretize_actions(actions_batch)
            
            logits = model(obs_batch)
            
            logits = logits.reshape(-1, num_bins)
            discrete_actions = discrete_actions.reshape(-1)
            
            loss = criterion(logits, discrete_actions)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_batch, actions_batch in val_loader:
                discrete_actions = model.discretize_actions(actions_batch)
                
                logits = model(obs_batch)
                
                logits = logits.reshape(-1, num_bins)
                discrete_actions = discrete_actions.reshape(-1)
                
                val_loss += criterion(logits, discrete_actions).item()
        
        val_loss /= len(val_loader)
        
        train_loss = np.mean(epoch_losses)
        losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            reward_sampling = evaluate_model(model, env, model_type='autoreg')
            reward_greedy = evaluate_model(model, env, model_type='autoreg', greedy=True)
            
            reward = max(reward_sampling, reward_greedy)
            
            eval_rewards.append(reward_sampling)
            eval_rewards_greedy.append(reward_greedy)
            eval_epochs.append(epoch + 1)
            
            print(f"Evaluation after epoch {epoch+1}:")
            print(f"  Sampling: Reward = {reward_sampling:.2f}")
            print(f"  Greedy: Reward = {reward_greedy:.2f}")
            
            if reward > best_reward:
                best_reward = reward
                best_model_state = {
                    'model_state': model.state_dict(),
                    'reward': reward,
                    'reward_sampling': reward_sampling,
                    'reward_greedy': reward_greedy,
                    'epoch': epoch + 1
                }
                torch.save(best_model_state, f'models/autoreg_best.pt')
                print(f"New best model saved with reward {reward:.2f}")
    
    training_time = time.time() - start_time
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state'])
    
    final_reward_sampling = evaluate_model(model, env, model_type='autoreg')
    final_reward_greedy = evaluate_model(model, env, model_type='autoreg', greedy=True)
    final_reward = max(final_reward_sampling, final_reward_greedy)
    
    print(f"Final evaluation rewards:")
    print(f"  Sampling: {final_reward_sampling:.2f}")
    print(f"  Greedy: {final_reward_greedy:.2f}")
    
    torch.save({
        'model_state': model.state_dict(),
        'reward_sampling': final_reward_sampling,
        'reward_greedy': final_reward_greedy,
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
        'num_bins': num_bins
    }, f'models/autoreg_final.pt')
    
    return {
        'model': model,
        'best_reward': best_reward,
        'final_reward': final_reward,
        'losses': losses,
        'val_losses': val_losses,
        'rewards': eval_rewards,
        'rewards_greedy': eval_rewards_greedy,
        'eval_epochs': eval_epochs,
        'training_time': training_time
    }

def main():
    hyperparams = {
        "lr": 1e-4,
        "batch_size": 2048,
        "epochs": 50,
        "eval_interval": 5,
        "num_bins": NUM_BINS
    }
    
    print("===== Training Autoregressive Discretized Model =====")
    model = AutoregressiveDiscretizedModel(env, num_bins=hyperparams["num_bins"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])
    
    results = train_autoreg_disc(
        model,
        observations,
        actions,
        batch_size=hyperparams["batch_size"],
        epochs=hyperparams["epochs"],
        lr=hyperparams["lr"],
        eval_interval=hyperparams["eval_interval"],
        env=env
    )
    
    safe_hyperparams = {}
    for key, value in hyperparams.items():
        if key not in ['env', 'device']:
            if hasattr(value, 'item'):
                safe_hyperparams[key] = value.item()
            else:
                safe_hyperparams[key] = value
                
    results = {
        "method": "AutoregressiveDiscretized",
        "timestamp": results["timestamp"],
        "final_reward_sampling": float(results["final_reward"]),
        "final_reward_greedy": float(results["final_reward"]),
        "best_reward": float(results["best_reward"]),
        "training_time_seconds": results["training_time"],
        "loss_history": [float(x) for x in results["losses"]],
        "val_loss_history": [float(x) for x in results["val_losses"]],
        "reward_history_sampling": [float(x) for x in results["rewards"]],
        "reward_history_greedy": [float(x) for x in results["rewards_greedy"]],
        "eval_epochs": results["eval_epochs"],
        "hyperparameters": safe_hyperparams
    }
    
    filename = f"results/autoreg_disc/autoreg_disc_{results['timestamp']}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main() 
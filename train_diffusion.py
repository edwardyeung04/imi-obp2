import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import gymnasium as gym
from actor_impl import Actor
from train import create_datasets, evaluate_model, device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data...")
data = torch.load('data/halfcheetah_v4_data.pt', weights_only=False)
observations = data['observations'].to(device)
actions = data['actions'].to(device)
print(f"Loaded {len(observations)} observation-action pairs")
print(f"Mean reward in dataset: {data['mean_reward']}, Std: {data['std_reward']}")

env = gym.make("HalfCheetah-v4")
env.single_observation_space = env.observation_space
env.single_action_space = env.action_space

class DiffusionPolicy(nn.Module):
    def __init__(self, env, hidden_dim=256, n_timesteps=100, beta_schedule='cosine'):
        super().__init__()
        self.obs_dim = np.array(env.single_observation_space.shape).prod()
        self.action_dim = np.prod(env.single_action_space.shape)
        
        self.fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.fc_noise1 = nn.Linear(hidden_dim + hidden_dim + self.action_dim, hidden_dim)
        self.fc_noise2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_noise_out = nn.Linear(hidden_dim, self.action_dim)
        
        self.n_timesteps = n_timesteps
        self.beta_schedule = beta_schedule
        self.set_beta_schedule(beta_schedule, n_timesteps)
        
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
    
    def set_beta_schedule(self, schedule_type, n_timesteps):
        if schedule_type == 'linear':
            self.beta = torch.linspace(1e-4, 0.02, n_timesteps).to(device)
        elif schedule_type == 'cosine':
            steps = n_timesteps + 1
            x = torch.linspace(0, n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / n_timesteps) + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = torch.clip(betas, 0, 0.999).to(device)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule_type}")
        
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def extract_features(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    def forward(self, obs, noisy_actions, t):
        obs_features = self.extract_features(obs)
        t_emb = self.time_embed(t.unsqueeze(-1))
        x = torch.cat([obs_features, t_emb, noisy_actions], dim=-1)
        x = F.relu(self.fc_noise1(x))
        x = F.relu(self.fc_noise2(x))
        predicted_noise = self.fc_noise_out(x)
        return predicted_noise
    
    def add_noise(self, actions, t):
        alpha_bar_t = self.alpha_bar[t]
        if len(alpha_bar_t.shape) == 0:
            alpha_bar_t = alpha_bar_t.unsqueeze(0)
        
        alpha_bar_t = alpha_bar_t.unsqueeze(-1).expand(-1, actions.shape[-1])
        noise = torch.randn_like(actions)
        noisy_actions = torch.sqrt(alpha_bar_t) * actions + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_actions, noise
    
    def sample(self, obs, steps=50, guidance_weight=0.0):
        batch_size = obs.shape[0]
        x = torch.randn(batch_size, self.action_dim).to(device)
        timesteps = torch.linspace(self.n_timesteps - 1, 0, steps).long().to(device)
        null_obs = torch.zeros_like(obs) if guidance_weight > 0 else None
        
        for i, timestep in enumerate(timesteps):
            t = torch.ones(batch_size, dtype=torch.long, device=device) * timestep
            t_scaled = t.float() / self.n_timesteps
            predicted_noise = self(obs, x, t_scaled)
            
            if guidance_weight > 0:
                predicted_noise_uncond = self(null_obs, x, t_scaled)
                predicted_noise = predicted_noise_uncond + guidance_weight * (predicted_noise - predicted_noise_uncond)
            
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            
            if i == len(timesteps) - 1:
                alpha_next = torch.ones_like(alpha_t)
                alpha_bar_next = torch.ones_like(alpha_bar_t)
            else:
                next_t = timesteps[i + 1]
                alpha_next = self.alpha[next_t]
                alpha_bar_next = self.alpha_bar[next_t]
            
            alpha_t = alpha_t.view(-1, 1)
            alpha_bar_t = alpha_bar_t.view(-1, 1)
            alpha_next = alpha_next.view(-1, 1)
            alpha_bar_next = alpha_bar_next.view(-1, 1)
            
            x_0_pred = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t + 1e-8)
            x_0_pred = torch.clamp(x_0_pred, -10, 10)
            x = torch.sqrt(alpha_bar_next) * x_0_pred + torch.sqrt(1 - alpha_bar_next) * predicted_noise
        
        final_pred = (x - torch.sqrt(1 - self.alpha_bar[0]) * predicted_noise) / torch.sqrt(self.alpha_bar[0] + 1e-8)
        final_pred = torch.clamp(final_pred, -10, 10)
        actions = torch.tanh(final_pred) * self.action_scale + self.action_bias
        return actions

def diffusion_to_actor(diffusion_model, actor_model):
    actor_model.fc1.weight.data.copy_(diffusion_model.fc1.weight.data)
    actor_model.fc1.bias.data.copy_(diffusion_model.fc1.bias.data)
    actor_model.fc2.weight.data.copy_(diffusion_model.fc2.weight.data)
    actor_model.fc2.bias.data.copy_(diffusion_model.fc2.bias.data)
    
    actor_model.fc_mean.weight.data.normal_(0, 0.01)
    actor_model.fc_mean.bias.data.zero_()
    actor_model.fc_logstd.weight.data.normal_(0, 0.01)
    actor_model.fc_logstd.bias.data.fill_(-2)
    
    return actor_model

def train_diffusion(model, observations, actions, **kwargs):
    batch_size = kwargs.get('batch_size', 4096)
    epochs = kwargs.get('epochs', 50)
    lr = kwargs.get('lr', 3e-4)
    eval_interval = kwargs.get('eval_interval', 5)
    env = kwargs.get('env', None)
    n_timesteps = kwargs.get('n_timesteps', 100)
    sampling_steps = kwargs.get('sampling_steps', 20)
    
    if env is None:
        from train import setup_env
        env = setup_env()
    
    if not isinstance(model, DiffusionPolicy):
        model = DiffusionPolicy(env, n_timesteps=n_timesteps).to(device)
    
    train_loader, val_loader = create_datasets(observations, actions, batch_size)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    val_losses = []
    eval_rewards = []
    eval_epochs = []
    best_reward = -float('inf')
    best_model_state = None
    
    ema_model = DiffusionPolicy(
        env, 
        hidden_dim=model.fc1.out_features,
        n_timesteps=n_timesteps,
        beta_schedule=model.beta_schedule
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_rate = 0.995
    
    print(f"Training diffusion model for {epochs} epochs with {n_timesteps} timesteps")
    print(f"Batch size: {batch_size}, Sampling steps: {sampling_steps}")
    print(f"Steps per epoch: {len(train_loader)}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for obs_batch, actions_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_size = obs_batch.shape[0]
            
            normalized_actions = (actions_batch - model.action_bias) / model.action_scale
            normalized_actions = torch.atanh(torch.clamp(normalized_actions, -0.999, 0.999))
            
            t = torch.randint(0, model.n_timesteps, (batch_size,), device=device)
            t_scaled = t.float() / model.n_timesteps
            
            noisy_actions, target_noise = model.add_noise(normalized_actions, t)
            
            predicted_noise = model(obs_batch, noisy_actions, t_scaled)
            
            loss = F.mse_loss(predicted_noise, target_noise)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data = ema_param.data * ema_rate + param.data * (1 - ema_rate)
            
            epoch_losses.append(loss.item())
        
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for obs_batch, actions_batch in val_loader:
                batch_size = obs_batch.shape[0]
                
                normalized_actions = (actions_batch - model.action_bias) / model.action_scale
                normalized_actions = torch.atanh(torch.clamp(normalized_actions, -0.999, 0.999))
                
                t = torch.randint(0, model.n_timesteps, (batch_size,), device=device)
                t_scaled = t.float() / model.n_timesteps
                
                noisy_actions, target_noise = model.add_noise(normalized_actions, t)
                
                predicted_noise = model(obs_batch, noisy_actions, t_scaled)
                
                val_loss += F.mse_loss(predicted_noise, target_noise).item()
                n_val_batches += 1
        
        if n_val_batches > 0:
            val_loss /= n_val_batches
        
        train_loss = np.mean(epoch_losses)
        losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            reward = evaluate_model(ema_model, env, model_type='diffusion')
            eval_rewards.append(reward)
            eval_epochs.append(epoch + 1)
            
            print(f"Evaluation after epoch {epoch+1}: Reward = {reward:.2f}")
            
            if reward > best_reward:
                best_reward = reward
                best_model_state = {
                    'model_state': ema_model.state_dict(),
                    'reward': reward,
                    'epoch': epoch + 1,
                    'n_timesteps': n_timesteps,
                    'sampling_steps': sampling_steps
                }
                torch.save(best_model_state, f'models/diffusion_best.pt')
                print(f"New best model saved with reward {reward:.2f}")
    
    training_time = time.time() - start_time
    
    if best_model_state is not None:
        ema_model.load_state_dict(best_model_state['model_state'])
    
    final_reward = evaluate_model(ema_model, env, model_type='diffusion')
    print(f"Final diffusion model evaluation: {final_reward:.2f}")
    
    torch.save({
        'model_state': ema_model.state_dict(),
        'reward': final_reward,
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
        'n_timesteps': n_timesteps,
        'sampling_steps': sampling_steps
    }, f'models/diffusion_final.pt')
    
    actor_model = Actor(env).to(device)
    actor_model = diffusion_to_actor(ema_model, actor_model)
    actor_reward = evaluate_model(actor_model, env)
    
    torch.save({
        'model_state': actor_model.state_dict(),
        'reward': actor_reward,
        'timestamp': time.strftime("%Y%m%d_%H%M%S")
    }, f'models/diffusion_actor_converted.pt')
    
    return {
        'model': ema_model,
        'actor_model': actor_model,
        'best_reward': best_reward,
        'final_reward': final_reward,
        'actor_reward': actor_reward,
        'losses': losses,
        'val_losses': val_losses,
        'rewards': eval_rewards,
        'eval_epochs': eval_epochs,
        'training_time': training_time
    }

def main():
    hyperparams = {
        "lr": 1e-4,
        "batch_size": 4096,
        "epochs": 50,
        "eval_interval": 5,
        "n_timesteps": 100,
        "beta_schedule": "cosine",
        "sampling_steps": 50
    }
    
    os.makedirs("models/diffusion", exist_ok=True)
    os.makedirs("results/diffusion", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    diffusion_model = DiffusionPolicy(
        env, 
        hidden_dim=256,
        n_timesteps=hyperparams["n_timesteps"],
        beta_schedule=hyperparams["beta_schedule"]
    ).to(device)
    
    optimizer = optim.Adam(diffusion_model.parameters(), lr=hyperparams["lr"])
    
    results = train_diffusion(
        diffusion_model, 
        observations, 
        actions, 
        batch_size=hyperparams["batch_size"], 
        epochs=hyperparams["epochs"],
        eval_interval=hyperparams["eval_interval"],
        n_timesteps=hyperparams["n_timesteps"],
        sampling_steps=hyperparams["sampling_steps"]
    )
    
    save_results_to_json(results['final_reward'], results['training_time'], results['losses'], results['rewards'], hyperparams)
    
    print(f"Final evaluation with diffusion model: {results['final_reward']:.2f}")
    print(f"Final evaluation with converted actor: {results['actor_reward']:.2f}")

if __name__ == "__main__":
    main() 
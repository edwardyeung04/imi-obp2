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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
print("Loading data...")
data = torch.load('data/halfcheetah_v4_data.pt', weights_only=False)
observations = data['observations'].to(device)
actions = data['actions'].to(device)
print(f"Loaded {len(observations)} observation-action pairs")
print(f"Mean reward in dataset: {data['mean_reward']}, Std: {data['std_reward']}")

# Create environment for actor model
env = gym.make("HalfCheetah-v4")
# Add necessary attributes to make it compatible with Actor class
env.single_observation_space = env.observation_space
env.single_action_space = env.action_space

# Enhanced Diffusion model for action prediction
class DiffusionPolicy(nn.Module):
    def __init__(self, env, hidden_dim=256, n_timesteps=100, beta_schedule='cosine'):
        super().__init__()
        self.obs_dim = np.array(env.single_observation_space.shape).prod()
        self.action_dim = np.prod(env.single_action_space.shape)
        
        # Feature extractor (same as original Actor for easier conversion)
        self.fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Time embedding layers
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Noise prediction network
        self.fc_noise1 = nn.Linear(hidden_dim + hidden_dim + self.action_dim, hidden_dim)
        self.fc_noise2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_noise_out = nn.Linear(hidden_dim, self.action_dim)
        
        # Diffusion process parameters
        self.n_timesteps = n_timesteps
        self.beta_schedule = beta_schedule
        self.set_beta_schedule(beta_schedule, n_timesteps)
        
        # For action scaling
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
        """Set up the noise schedule based on different options"""
        if schedule_type == 'linear':
            self.beta = torch.linspace(1e-4, 0.02, n_timesteps).to(device)
        elif schedule_type == 'cosine':
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
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
        """Extract observation features"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    def forward(self, obs, noisy_actions, t):
        """Predict noise given noisy actions, observation, and timestep"""
        # Feature extraction
        obs_features = self.extract_features(obs)
        
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))
        
        # Concat inputs
        x = torch.cat([obs_features, t_emb, noisy_actions], dim=-1)
        
        # Process through noise prediction network
        x = F.relu(self.fc_noise1(x))
        x = F.relu(self.fc_noise2(x))
        predicted_noise = self.fc_noise_out(x)
        
        return predicted_noise
    
    def add_noise(self, actions, t):
        """Add noise to actions at timestep t"""
        # Get the corresponding alpha_bar value
        alpha_bar_t = self.alpha_bar[t]
        if len(alpha_bar_t.shape) == 0:
            alpha_bar_t = alpha_bar_t.unsqueeze(0)
        
        # Expand to match batch dimension
        alpha_bar_t = alpha_bar_t.unsqueeze(-1).expand(-1, actions.shape[-1])
        
        # Sample noise
        noise = torch.randn_like(actions)
        
        # Add noise according to diffusion process
        noisy_actions = torch.sqrt(alpha_bar_t) * actions + torch.sqrt(1 - alpha_bar_t) * noise
        
        return noisy_actions, noise
    
    def sample(self, obs, steps=50, guidance_weight=0.0):
        """Sample actions using DDIM sampling with optional classifier-free guidance"""
        batch_size = obs.shape[0]
        
        # Start with random noise
        x = torch.randn(batch_size, self.action_dim).to(device)
        
        # Set up timestep sequence for sampling
        timesteps = torch.linspace(self.n_timesteps - 1, 0, steps).long().to(device)
        
        # For classifier-free guidance
        null_obs = torch.zeros_like(obs) if guidance_weight > 0 else None
        
        # Iteratively denoise
        for i, timestep in enumerate(timesteps):
            t = torch.ones(batch_size, dtype=torch.long, device=device) * timestep
            
            # Scale timestep to [0, 1] for model input
            t_scaled = t.float() / self.n_timesteps
            
            # Predict noise
            predicted_noise = self(obs, x, t_scaled)
            
            # Apply classifier-free guidance if enabled
            if guidance_weight > 0:
                predicted_noise_uncond = self(null_obs, x, t_scaled)
                predicted_noise = predicted_noise_uncond + guidance_weight * (predicted_noise - predicted_noise_uncond)
            
            # Get alpha values for current and previous timestep
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            
            if i == len(timesteps) - 1:
                # Last step
                alpha_next = torch.ones_like(alpha_t)
                alpha_bar_next = torch.ones_like(alpha_bar_t)
            else:
                next_t = timesteps[i + 1]
                alpha_next = self.alpha[next_t]
                alpha_bar_next = self.alpha_bar[next_t]
            
            # Expand alpha dimensions to match actions
            alpha_t = alpha_t.view(-1, 1)
            alpha_bar_t = alpha_bar_t.view(-1, 1)
            alpha_next = alpha_next.view(-1, 1)
            alpha_bar_next = alpha_bar_next.view(-1, 1)
            
            # DDIM update with improved numerical stability
            x_0_pred = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t + 1e-8)
            x_0_pred = torch.clamp(x_0_pred, -10, 10)  # Prevent extreme values
            
            # DDIM sampling
            x = torch.sqrt(alpha_bar_next) * x_0_pred + torch.sqrt(1 - alpha_bar_next) * predicted_noise
        
        # Apply final noise prediction to get clean prediction
        final_pred = (x - torch.sqrt(1 - self.alpha_bar[0]) * predicted_noise) / torch.sqrt(self.alpha_bar[0] + 1e-8)
        final_pred = torch.clamp(final_pred, -10, 10)
        
        # Scale actions to valid range
        actions = torch.tanh(final_pred) * self.action_scale + self.action_bias
        return actions

# Improved conversion from Diffusion Policy to Actor
def diffusion_to_actor(diffusion_model, actor_model):
    """Maps the diffusion parameters to standard Actor parameters where possible"""
    # Get state dicts
    diffusion_state_dict = diffusion_model.state_dict()
    actor_state_dict = actor_model.state_dict()
    
    # Copy over shared parameters
    for key in actor_state_dict.keys():
        if key in diffusion_state_dict:
            actor_state_dict[key] = diffusion_state_dict[key]
    
    # Load the state dict
    actor_model.load_state_dict(actor_state_dict)
    return actor_model

# Function to evaluate the model in the environment
def evaluate_model(model, num_episodes=10, use_diffusion=True):
    """
    Evaluate the policy in the environment
    """
    rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            with torch.no_grad():
                # Use the diffusion model for sampling actions
                if use_diffusion:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = model.sample(obs_tensor).cpu().squeeze().numpy()
                else:
                    # Using converted actor
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    _, _, action = model.get_action(obs_tensor)
                    action = action.cpu().squeeze().numpy()
            
            # Step in the environment
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
        rewards.append(episode_reward)
        
    mean_reward = np.mean(rewards)
    return mean_reward

# Save results for reporting
def save_results_to_json(final_reward, training_time, losses, rewards, hyperparams=None):
    """Save training results to a JSON file for later analysis and reporting."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if hyperparams is None:
        hyperparams = {}
        
    results = {
        "method": "Diffusion",
        "timestamp": timestamp,
        "final_reward": float(final_reward),
        "training_time_seconds": training_time,
        "loss_history": [float(x) for x in losses],
        "reward_history": [float(x) for x in rewards],
        "hyperparameters": {
            "learning_rate": hyperparams.get("lr", 3e-4),
            "batch_size": hyperparams.get("batch_size", 4096),
            "epochs": hyperparams.get("epochs", 50),
            "eval_interval": hyperparams.get("eval_interval", 5),
            "n_timesteps": hyperparams.get("n_timesteps", 100),
            "beta_schedule": hyperparams.get("beta_schedule", "cosine"),
            "sampling_steps": hyperparams.get("sampling_steps", 50)
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs("results/diffusion", exist_ok=True)
    
    # Save results
    filename = f"results/diffusion/diffusion_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename

# Training function with EMA for stability
def train(model, optimizer, batch_size=4096, epochs=50, eval_interval=5, n_timesteps=100):
    dataset_size = len(observations)
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * epochs
    
    # Create EMA version of model for more stable evaluation
    ema_model = DiffusionPolicy(
        env, 
        hidden_dim=model.fc1.out_features,
        n_timesteps=n_timesteps,
        beta_schedule=model.beta_schedule
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_rate = 0.995
    
    losses = []
    eval_rewards = []
    best_reward = -float('inf')
    best_model_state = None
    
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle data for each epoch
        indices = torch.randperm(dataset_size)
        obs_shuffled = observations[indices]
        actions_shuffled = actions[indices]
        
        # Set model to train mode
        model.train()
        
        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}"):
            # Get batch
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            obs_batch = obs_shuffled[start_idx:end_idx]
            actions_batch = actions_shuffled[start_idx:end_idx]
            
            # Normalize actions to [-1, 1] range before diffusion
            normalized_actions = (actions_batch - model.action_bias) / model.action_scale
            normalized_actions = torch.atanh(torch.clamp(normalized_actions, -0.999, 0.999))
            
            # Sample timesteps
            t = torch.randint(0, n_timesteps, (batch_size,), device=device)
            t_scaled = t.float() / n_timesteps
            
            # Add noise to actions
            noisy_actions, noise = model.add_noise(normalized_actions, t)
            
            # Predict noise
            predicted_noise = model(obs_batch, noisy_actions, t_scaled)
            
            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Update EMA model
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data = ema_param.data * ema_rate + param.data * (1 - ema_rate)
            
            epoch_losses.append(loss.item())
        
        # Record average loss for the epoch
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate model
        if (epoch + 1) % eval_interval == 0:
            # Set to eval mode for evaluation
            ema_model.eval()
            reward = evaluate_model(ema_model, num_episodes=5)
            eval_rewards.append(reward)
            print(f"Evaluation after epoch {epoch+1}: Reward = {reward:.2f}")
            
            # Save best model
            if reward > best_reward:
                best_reward = reward
                best_model_state = {
                    'model_state': ema_model.state_dict(),
                    'reward': reward,
                    'epoch': epoch + 1
                }
                model_path = f'models/diffusion_best.pt'
                os.makedirs('models', exist_ok=True)
                torch.save(best_model_state, model_path)
                print(f"New best model saved with reward {best_reward:.2f}")
    
    # Training complete
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best reward achieved: {best_reward:.2f}")
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state'])
        ema_model.load_state_dict(best_model_state['model_state'])
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(list(range(eval_interval, epochs + 1, eval_interval)), eval_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('plots', exist_ok=True)
    plot_path = f'plots/diffusion_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    
    return best_reward, losses, eval_rewards, training_time, ema_model

def main():
    # Hyperparameters matched with MoG implementation for fair comparison
    hyperparams = {
        "lr": 3e-4,                # Match MoG learning rate
        "batch_size": 4096,        # Match MoG batch size
        "epochs": 50,              # Match MoG epochs
        "eval_interval": 5,        # Match MoG evaluation interval
        "n_timesteps": 100,        # Keep reasonable diffusion timesteps
        "beta_schedule": "cosine", # Use cosine schedule for better results
        "sampling_steps": 50       # Reasonable number of sampling steps
    }
    
    # Create directories if they don't exist
    os.makedirs("models/diffusion", exist_ok=True)
    os.makedirs("results/diffusion", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Create model and optimizer
    diffusion_model = DiffusionPolicy(
        env, 
        hidden_dim=256,  # Match MoG hidden dim
        n_timesteps=hyperparams["n_timesteps"],
        beta_schedule=hyperparams["beta_schedule"]
    ).to(device)
    
    optimizer = optim.Adam(diffusion_model.parameters(), lr=hyperparams["lr"])
    
    # Train model
    best_reward, losses, eval_rewards, training_time, ema_model = train(
        diffusion_model, 
        optimizer, 
        batch_size=hyperparams["batch_size"], 
        epochs=hyperparams["epochs"],
        eval_interval=hyperparams["eval_interval"],
        n_timesteps=hyperparams["n_timesteps"]
    )
    
    # Convert to standard actor for final evaluation
    actor = Actor(env).to(device)
    converted_actor = diffusion_to_actor(ema_model, actor)
    
    # Evaluate both models
    diffusion_reward = evaluate_model(ema_model, num_episodes=20, use_diffusion=True)
    actor_reward = evaluate_model(converted_actor, num_episodes=20, use_diffusion=False)
    
    print(f"Final evaluation with diffusion model: {diffusion_reward:.2f}")
    print(f"Final evaluation with converted actor: {actor_reward:.2f}")
    
    # Save the final models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diffusion_path = f'models/diffusion/diffusion_{timestamp}.pt'
    actor_path = f'models/diffusion/actor_from_diffusion_{timestamp}.pt'
    
    os.makedirs('models/diffusion', exist_ok=True)
    
    torch.save({
        'model_state': ema_model.state_dict(),
        'reward': diffusion_reward,
        'hyperparams': hyperparams
    }, diffusion_path)
    
    torch.save({
        'model_state': converted_actor.state_dict(),
        'reward': actor_reward
    }, actor_path)
    
    # Save results
    save_results_to_json(diffusion_reward, training_time, losses, eval_rewards, hyperparams)
    
    print(f"Diffusion model saved to {diffusion_path}")
    print(f"Converted actor saved to {actor_path}")

if __name__ == "__main__":
    main() 
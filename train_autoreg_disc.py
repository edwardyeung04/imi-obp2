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

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results/autoreg_disc', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
print("Loading data...")
data = torch.load('data/halfcheetah_v4_data.pt', map_location=device, weights_only=False)
observations = data['observations'].to(device)
actions = data['actions'].to(device)
print(f"Loaded {len(observations)} observation-action pairs")
print(f"Mean reward in dataset: {data['mean_reward']}, Std: {data['std_reward']}")

# Create environment for actor model
env = gym.make("HalfCheetah-v4")
# Add necessary attributes to make it compatible with Actor class
env.single_observation_space = env.observation_space
env.single_action_space = env.action_space

# Parameters for discretization
NUM_BINS = 21  # Number of bins for discretizing each action dimension
ACTION_DIM = np.prod(env.single_action_space.shape).item()  # Number of action dimensions

class AutoregressiveDiscretizedModel(nn.Module):
    def __init__(self, env, num_bins=NUM_BINS, hidden_dim=256):
        super().__init__()
        self.obs_dim = np.array(env.single_observation_space.shape).prod()
        self.action_dim = np.prod(env.single_action_space.shape).item()
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        
        # Base feature extractor
        self.shared_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Simple autoregressive network
        # We'll predict each action dimension one at a time
        # Input: obs_features + previous_actions
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim * num_bins)  # Output logits for all actions
        )
        
        # action rescaling (same as Actor)
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
        
        # Bin centers for converting from discrete to continuous
        bin_edges = torch.linspace(-1, 1, num_bins + 1)  # Create bin edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Get centers of bins
        self.register_buffer("bin_centers", bin_centers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def discretize_actions(self, actions):
        """Convert continuous actions to discrete bin indices"""
        # Normalize actions to [-1, 1] range
        normalized_actions = (actions - self.action_bias) / self.action_scale
        normalized_actions = torch.clamp(normalized_actions, -1, 1)
        
        # Convert to bin indices
        # Scale to [0, num_bins-1] range
        indices = ((normalized_actions + 1) / 2 * (self.num_bins - 1)).long()
        indices = torch.clamp(indices, 0, self.num_bins - 1)
        
        return indices
    
    def continuous_from_discrete(self, discrete_actions):
        """Convert discrete bin indices back to continuous actions"""
        # Get the bin centers for each discrete action
        continuous_normalized = self.bin_centers[discrete_actions]
        
        # Convert back to original action range
        continuous_actions = continuous_normalized * self.action_scale + self.action_bias
        
        return continuous_actions
    
    def forward(self, obs):
        """
        Forward pass for training - returns logits for all actions
        """
        batch_size = obs.shape[0]
        
        # Generate base encoding
        base_features = self.shared_encoder(obs)
        
        # For training, we'll use teacher forcing
        # Pass zeros as previous actions (will be ignored since we'll use a mask)
        prev_actions_placeholder = torch.zeros(
            batch_size, self.action_dim, device=obs.device
        )
        
        # Concatenate features with dummy previous actions
        combined = torch.cat([base_features, prev_actions_placeholder], dim=1)
        
        # Get logits for all action dimensions
        all_logits = self.action_predictor(combined)
        
        # Reshape logits to [batch_size, action_dim, num_bins]
        all_logits = all_logits.view(batch_size, self.action_dim, self.num_bins)
        
        return all_logits
    
    def get_action(self, obs, greedy=False):
        """
        Autoregressively generate actions for inference
        
        Args:
            obs: Batch of observations [batch_size, obs_dim]
            greedy: If True, take the most likely action; otherwise sample
            
        Returns:
            Continuous actions [batch_size, action_dim]
        """
        batch_size = obs.shape[0]
        self.eval()
        
        with torch.no_grad():
            # Generate base encoding
            base_features = self.shared_encoder(obs)
            
            # Initialize tensor to store the discrete actions
            discrete_actions = torch.zeros(
                batch_size, self.action_dim, dtype=torch.long, device=obs.device
            )
            
            # Placeholder for previous predictions (used by autoregressive model)
            prev_actions = torch.zeros(
                batch_size, self.action_dim, device=obs.device
            )
            
            # Predict each action dimension one at a time
            for i in range(self.action_dim):
                # Concatenate base features with current placeholder
                combined = torch.cat([base_features, prev_actions], dim=1)
                
                # Get logits for all dimensions
                all_logits = self.action_predictor(combined)
                all_logits = all_logits.view(batch_size, self.action_dim, self.num_bins)
                
                # Get logits for the current dimension only
                current_logits = all_logits[:, i, :]
                
                # Sample or take argmax
                if greedy:
                    current_action = torch.argmax(current_logits, dim=1)
                else:
                    probs = F.softmax(current_logits, dim=1)
                    current_action = torch.multinomial(probs, 1).squeeze(-1)
                
                # Update discrete actions with the new sample
                discrete_actions[:, i] = current_action
                
                # Update placeholder for next iteration if needed
                if i < self.action_dim - 1:
                    # Just use the bin center as the normalized value
                    prev_actions[:, i] = self.bin_centers[current_action]
            
            # Convert full discrete action tensor to continuous
            continuous_actions = self.continuous_from_discrete(discrete_actions)
            
            return continuous_actions

# Function to evaluate the model in the environment
def evaluate_model(model, num_episodes=10, greedy=False):
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not (done or truncated):
            with torch.no_grad():
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                
                # Get action
                action = model.get_action(obs_tensor, greedy=greedy)
                action = action.cpu().squeeze().numpy()
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)

# Training function
def train(model, optimizer, batch_size=4096, epochs=50, eval_interval=5):
    # Create dataset and dataloader
    dataset = TensorDataset(observations, actions)
    
    # Split into train and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)
    
    losses = []
    val_losses = []
    eval_rewards = []
    eval_rewards_greedy = []
    eval_epochs = []
    best_reward = -float('inf')
    best_model_state = None
    
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    print(f"Steps per epoch: {len(train_loader)}")
    
    start_time = time.time()
    
    # Cross entropy loss for classification
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for obs_batch, actions_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Discretize the continuous actions
            discrete_actions = model.discretize_actions(actions_batch)
            
            # Forward pass
            logits = model(obs_batch)  # [batch_size, action_dim, num_bins]
            
            # Reshape for cross entropy loss
            logits = logits.reshape(-1, NUM_BINS)  # [batch_size * action_dim, num_bins]
            discrete_actions = discrete_actions.reshape(-1)  # [batch_size * action_dim]
            
            # Calculate loss
            loss = criterion(logits, discrete_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Validate
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for obs_batch, actions_batch in val_loader:
                # Discretize the continuous actions
                discrete_actions = model.discretize_actions(actions_batch)
                
                # Forward pass
                logits = model(obs_batch)
                
                # Reshape for cross entropy loss
                logits = logits.reshape(-1, NUM_BINS)
                discrete_actions = discrete_actions.reshape(-1)
                
                # Calculate loss
                val_loss = criterion(logits, discrete_actions)
                epoch_val_losses.append(val_loss.item())
        
        # Record average loss for the epoch
        train_loss = np.mean(epoch_losses)
        val_loss = np.mean(epoch_val_losses)
        losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Evaluate model in the environment
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            # Evaluate with sampling
            reward = evaluate_model(model, num_episodes=5, greedy=False)
            # Evaluate with greedy actions
            reward_greedy = evaluate_model(model, num_episodes=5, greedy=True)
            
            eval_rewards.append(reward)
            eval_rewards_greedy.append(reward_greedy)
            eval_epochs.append(epoch + 1)
            
            print(f"Evaluation after epoch {epoch+1}:")
            print(f"  Sampling: Reward = {reward:.2f}")
            print(f"  Greedy: Reward = {reward_greedy:.2f}")
            
            # Save best model based on the best of sampling and greedy
            curr_best = max(reward, reward_greedy)
            if curr_best > best_reward:
                best_reward = curr_best
                best_model_state = {
                    'model_state': model.state_dict(),
                    'reward': curr_best,
                    'reward_sampling': reward,
                    'reward_greedy': reward_greedy,
                    'epoch': epoch + 1
                }
                model_path = f'models/autoreg_disc_best.pt'
                torch.save(best_model_state, model_path)
                print(f"New best model saved with reward {best_reward:.2f}")
    
    # Training complete
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best reward achieved: {best_reward:.2f}")
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs+1), losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(eval_epochs, eval_rewards, label='Sampling')
    plt.plot(eval_epochs, eval_rewards_greedy, label='Greedy')
    plt.title('Evaluation Rewards')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(eval_epochs, eval_rewards, label='Sampling')
    plt.title('Evaluation Rewards (Sampling)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    if len(eval_rewards) > 0:
        plt.ylim(max(0, min(eval_rewards) - 1000), max(eval_rewards) + 1000)  # Zoomed in view
    
    plt.subplot(2, 2, 4)
    plt.plot(eval_epochs, eval_rewards_greedy, label='Greedy')
    plt.title('Evaluation Rewards (Greedy)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    if len(eval_rewards_greedy) > 0:
        plt.ylim(max(0, min(eval_rewards_greedy) - 1000), max(eval_rewards_greedy) + 1000)  # Zoomed in view
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/autoreg_disc_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Load best model for return
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state'])
    
    return model, best_reward, losses, val_losses, eval_rewards, eval_rewards_greedy, eval_epochs, training_time

def main():
    # Hyperparameters
    hyperparams = {
        "lr": 1e-4,           # Learning rate
        "batch_size": 2048,    # Batch size
        "epochs": 50,          # Number of epochs
        "eval_interval": 5,    # Evaluate every N epochs
        "num_bins": NUM_BINS   # Number of bins for discretization
    }
    
    # Create the model
    print("===== Training Autoregressive Discretized Model =====")
    model = AutoregressiveDiscretizedModel(env, num_bins=hyperparams["num_bins"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])
    
    # Train the model
    model, best_reward, losses, val_losses, eval_rewards, eval_rewards_greedy, eval_epochs, training_time = train(
        model,
        optimizer,
        batch_size=hyperparams["batch_size"],
        epochs=hyperparams["epochs"],
        eval_interval=hyperparams["eval_interval"]
    )
    
    # Final evaluation with more episodes
    final_reward_sampling = evaluate_model(model, num_episodes=10, greedy=False)
    final_reward_greedy = evaluate_model(model, num_episodes=10, greedy=True)
    print(f"Final evaluation rewards:")
    print(f"  Sampling: {final_reward_sampling:.2f}")
    print(f"  Greedy: {final_reward_greedy:.2f}")
    
    # Save the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_state = {
        'model_state': model.state_dict(),
        'reward_sampling': final_reward_sampling,
        'reward_greedy': final_reward_greedy,
        'best_reward': best_reward,
        'timestamp': timestamp,
        'num_bins': NUM_BINS
    }
    
    torch.save(final_model_state, f'models/autoreg_disc_final_{timestamp}.pt')
    print(f"Final model saved to models/autoreg_disc_final_{timestamp}.pt")
    
    # Save results for reporting
    results = {
        "method": "AutoregressiveDiscretized",
        "timestamp": timestamp,
        "final_reward_sampling": float(final_reward_sampling),
        "final_reward_greedy": float(final_reward_greedy),
        "best_reward": float(best_reward),
        "training_time_seconds": training_time,
        "loss_history": [float(x) for x in losses],
        "val_loss_history": [float(x) for x in val_losses],
        "reward_history_sampling": [float(x) for x in eval_rewards],
        "reward_history_greedy": [float(x) for x in eval_rewards_greedy],
        "eval_epochs": eval_epochs,
        "hyperparameters": hyperparams
    }
    
    filename = f"results/autoreg_disc/autoreg_disc_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main() 
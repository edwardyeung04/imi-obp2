import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from actor_impl import Actor
import time
import os
import json
from tqdm import tqdm
from datetime import datetime

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results/mog', exist_ok=True)
os.makedirs('plots', exist_ok=True)

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

# Define Mixture of Gaussians Actor model
class MoGActor(nn.Module):
    def __init__(self, env, num_components=5):
        super().__init__()
        self.num_components = num_components
        self.obs_dim = np.array(env.single_observation_space.shape).prod()
        self.action_dim = np.prod(env.single_action_space.shape)
        
        # Shared feature extractor (match exact names with Actor implementation)
        self.fc1 = nn.Linear(self.obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Outputs for each Gaussian component
        self.means_layer = nn.Linear(256, self.action_dim * num_components)
        self.log_stds_layer = nn.Linear(256, self.action_dim * num_components)
        self.mixture_weights_layer = nn.Linear(256, num_components)
        
        # Also create standard outputs for behavioral cloning (match Actor impl)
        self.fc_mean = nn.Linear(256, self.action_dim)
        self.fc_logstd = nn.Linear(256, self.action_dim)
        
        # action rescaling
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
        
        # Initialize weights with small values to improve stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with smaller values for better numerical stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x):
        """Get features using the same structure as Actor"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    def forward(self, x):
        # Extract features
        features = self.extract_features(x)
        
        # Standard Actor outputs for behavioral cloning
        mean = self.fc_mean(features)
        log_std = self.fc_logstd(features)
        log_std = torch.tanh(log_std)
        log_std = -5 + 0.5 * (2 - (-5)) * (log_std + 1)  # same as Actor
        
        # Get mixture parameters
        mog_means = self.means_layer(features)
        mog_log_stds = self.log_stds_layer(features)
        mixture_logits = self.mixture_weights_layer(features)
        
        # Reshape for multiple components
        batch_size = x.shape[0]
        mog_means = mog_means.view(batch_size, self.num_components, self.action_dim)
        mog_log_stds = mog_log_stds.view(batch_size, self.num_components, self.action_dim)
        
        # Apply constraints
        mog_log_stds = torch.clamp(mog_log_stds, -5, 2)  # Same as original Actor
        
        # Apply softmax with numerical stability
        # Add small epsilon to avoid zero probabilities
        mixture_weights = F.softmax(mixture_logits, dim=-1)
        # Ensure mixture weights are valid probabilities
        mixture_weights = torch.clamp(mixture_weights, min=1e-6, max=1.0)
        mixture_weights = mixture_weights / mixture_weights.sum(dim=-1, keepdim=True)
        
        return mean, log_std, mog_means, mog_log_stds, mixture_weights
    
    def get_action(self, x):
        mean, log_std, mog_means, mog_log_stds, mixture_weights = self(x)
        batch_size = x.shape[0]
        
        # For normal Actor behavior (behavioral cloning part)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        bc_action = y_t * self.action_scale + self.action_bias
        
        # Compute standard log prob for behavioral cloning
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # For MoG - only used during training for log prob calculation
        if self.training:
            # During training: Sample component indices based on mixture weights
            # Add stability check
            if torch.isnan(mixture_weights).any() or torch.isinf(mixture_weights).any() or (mixture_weights < 0).any():
                # Handle numerical instability - use uniform distribution as fallback
                print("Warning: Invalid mixture weights detected. Using uniform distribution.")
                mixture_weights = torch.ones_like(mixture_weights) / self.num_components
            
            component_indices = torch.multinomial(mixture_weights, 1).squeeze(-1)
            
            # Select means and log_stds for the sampled components
            batch_indices = torch.arange(batch_size, device=x.device)
            selected_means = mog_means[batch_indices, component_indices]
            selected_log_stds = mog_log_stds[batch_indices, component_indices]
            
            # Sample from the selected components
            mog_std = torch.exp(selected_log_stds)
            mog_normal = torch.distributions.Normal(selected_means, mog_std)
            mog_x_t = mog_normal.rsample()  # reparameterization trick
            mog_y_t = torch.tanh(mog_x_t)
            mog_action = mog_y_t * self.action_scale + self.action_bias
            
            # Compute log probs (mixture of gaussians log prob)
            component_log_probs = []
            for c in range(self.num_components):
                # Compute log prob for each component
                std_c = torch.exp(mog_log_stds[:, c])
                normal_c = torch.distributions.Normal(mog_means[:, c], std_c)
                log_prob_c = normal_c.log_prob(mog_x_t)
                # Enforce action bounds
                log_prob_c -= torch.log(self.action_scale * (1 - mog_y_t.pow(2)) + 1e-6)
                log_prob_c = log_prob_c.sum(1, keepdim=True)
                component_log_probs.append(log_prob_c)
            
            # Stack and add mixture weights
            component_log_probs = torch.stack(component_log_probs, dim=-1)  # [batch, 1, n_components]
            # Add small epsilon to avoid log(0)
            safe_mixture_weights = torch.clamp(mixture_weights, min=1e-8)
            log_probs_weighted = component_log_probs + torch.log(safe_mixture_weights).unsqueeze(1)
            
            # Use log-sum-exp trick for numerical stability
            mog_log_prob = torch.logsumexp(log_probs_weighted, dim=-1)
            
            # Add a final stability check
            if torch.isnan(mog_log_prob).any() or torch.isinf(mog_log_prob).any():
                # If we still have NaNs, return a fixed log prob
                print("Warning: NaN or Inf in log prob. Using fixed value.")
                mog_log_prob = torch.ones_like(mog_log_prob) * -10.0
            
            # Return both log probs
            return bc_action, [log_prob, mog_log_prob], mean
        else:
            # During evaluation: Use standard Actor behavior (just using mean)
            # This matches the original request to "use the mean output rather than 
            # the sampled action for consistency in evals"
            mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
            return mean_action, log_prob, mean_action

# Function to convert MoG model to standard Actor model for evaluation
def mog_to_actor(mog_model, actor_model):
    """Maps the MoG parameters to standard Actor parameters"""
    # This is now much simpler since we use the same parameter names
    # Get state dicts
    mog_state_dict = mog_model.state_dict()
    actor_state_dict = actor_model.state_dict()
    
    # Copy over shared parameters (direct mapping)
    for key in actor_state_dict.keys():
        if key in mog_state_dict:
            actor_state_dict[key] = mog_state_dict[key]
    
    # Load the state dict
    actor_model.load_state_dict(actor_state_dict)
    return actor_model

# Function to evaluate the model in the environment
def evaluate_model(model, num_episodes=10, use_mog=True):
    """
    Evaluates the model in the environment.
    If use_mog is True, directly uses the MoG model.
    If use_mog is False, converts MoG to standard Actor model first.
    """
    if not use_mog:
        # Convert MoG to standard Actor for evaluation
        actor = Actor(env).to(device)
        actor = mog_to_actor(model, actor)
    else:
        actor = model  # Use MoG model directly
        
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not (done or truncated):
            with torch.no_grad():
                # Use mean action for evaluation as specified
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                _, _, action = actor.get_action(obs_tensor)
                action = action.cpu().squeeze().numpy()
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)

# Save results for reporting
def save_results_to_json(num_components, final_reward, training_time, losses, rewards, hyperparams=None):
    """Save training results to a JSON file for later analysis and reporting."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if hyperparams is None:
        hyperparams = {}
        
    results = {
        "method": f"MoG-{num_components}",
        "timestamp": timestamp,
        "final_reward": float(final_reward),
        "training_time_seconds": training_time,
        "loss_history": [float(x) for x in losses],
        "reward_history": [float(x) for x in rewards],
        "hyperparameters": {
            "num_components": num_components,
            "learning_rate": hyperparams.get("lr", 3e-4),
            "batch_size": hyperparams.get("batch_size", 4096),
            "epochs": hyperparams.get("epochs", 50),
            "eval_interval": hyperparams.get("eval_interval", 5),
            "bc_weight": hyperparams.get("bc_weight", 0.5)
        }
    }
    
    # Save results
    filename = f"results/mog/mog_{num_components}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename

# Compute MSE loss between predicted and expert actions
def compute_bc_loss(predicted_actions, expert_actions):
    return F.mse_loss(predicted_actions, expert_actions)

# Training function - add behavioral cloning
def train(model, optimizer, batch_size=1024, epochs=100, eval_interval=5, num_components=5, bc_weight=0.5):
    dataset_size = len(observations)
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * epochs
    
    losses = []
    eval_rewards = []
    best_reward = -float('inf')
    best_model_state = None
    
    # Set gradient clipping threshold
    max_grad_norm = 1.0
    
    print(f"Training MoG with {num_components} components and BC weight {bc_weight}")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    print(f"Using gradient clipping with max norm: {max_grad_norm}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle data for each epoch
        indices = torch.randperm(dataset_size)
        obs_shuffled = observations[indices]
        actions_shuffled = actions[indices]
        
        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}"):
            # Get batch
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            obs_batch = obs_shuffled[start_idx:end_idx]
            actions_batch = actions_shuffled[start_idx:end_idx]
            
            # Forward pass - make sure model is in training mode
            model.train()
            predicted_actions, log_probs, _ = model.get_action(obs_batch)
            
            # Unpack log probs
            bc_log_prob, mog_log_prob = log_probs
            
            # Compute behavioral cloning loss (MSE)
            bc_loss = compute_bc_loss(predicted_actions, actions_batch)
            
            # Compute negative log probability losses
            neg_bc_log_prob = -bc_log_prob.mean()
            neg_mog_log_prob = -mog_log_prob.mean()
            
            # Combine losses with weighting
            # BC weight controls balance between direct action prediction and log prob
            loss = bc_weight * bc_loss + (1-bc_weight) * neg_mog_log_prob
            
            # Debug info
            if step == 0 and epoch % 5 == 0:
                print(f"  BC Loss: {bc_loss:.4f}, NegLogProb: {neg_mog_log_prob:.4f}, Combined: {loss:.4f}")
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss encountered at step {step}. Skipping step.")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Periodically check for NaNs in model parameters
            if step % 100 == 0:
                has_nan = False
                for name, param in model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        has_nan = True
                        print(f"Warning: NaN or Inf found in {name}")
                
                if has_nan:
                    print("Resetting optimizer state due to NaNs")
                    optimizer = optim.Adam(model.parameters(), lr=3e-4)
        
        # Record average loss for the epoch
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate model
        if (epoch + 1) % eval_interval == 0:
            model.eval()
            reward = evaluate_model(model, use_mog=True)
            eval_rewards.append(reward)
            print(f"Evaluation after epoch {epoch+1}: Reward = {reward:.2f}")
            
            # Save best model
            if reward > best_reward:
                best_reward = reward
                best_model_state = {
                    'model_state': model.state_dict(),
                    'num_components': num_components,
                    'reward': reward,
                    'epoch': epoch + 1,
                    'bc_weight': bc_weight
                }
                model_path = f'models/mog_{num_components}_best.pt'
                torch.save(best_model_state, model_path)
                
                # Also save an Actor-compatible version
                actor = Actor(env).to(device)
                actor = mog_to_actor(model, actor)
                actor_path = f'models/mog_{num_components}_actor_converted.pt'
                torch.save(actor.state_dict(), actor_path)
                
                print(f"New best model saved with reward {best_reward:.2f}")
    
    # Training complete
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best reward achieved: {best_reward:.2f}")
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state'])
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(list(range(eval_interval, epochs + 1, eval_interval)), eval_rewards)
    plt.title(f'Evaluation Rewards (MoG, {num_components} components)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/mog_{num_components}_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    
    return best_reward, losses, eval_rewards, training_time

def main():
    # Hyperparameters
    hyperparams = {
        "lr": 3e-4,            # Increased back from 1e-4
        "batch_size": 4096,
        "epochs": 50,
        "eval_interval": 5,
        "num_components": 5,
        "bc_weight": 0.5       # Weight for behavioral cloning loss
    }
    
    # Create model and optimizer
    model = MoGActor(env, num_components=hyperparams["num_components"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])
    
    # Train model
    best_reward, losses, eval_rewards, training_time = train(
        model, 
        optimizer, 
        batch_size=hyperparams["batch_size"], 
        epochs=hyperparams["epochs"],
        eval_interval=hyperparams["eval_interval"],
        num_components=hyperparams["num_components"],
        bc_weight=hyperparams["bc_weight"]
    )
    
    # Final evaluation with the MoG model
    model.eval()
    final_reward_mog = evaluate_model(model, num_episodes=20, use_mog=True)
    print(f"Final evaluation reward (MoG): {final_reward_mog:.2f}")
    
    # Convert to standard Actor model and evaluate
    actor = Actor(env).to(device)
    actor = mog_to_actor(model, actor)
    final_reward_actor = evaluate_model(actor, num_episodes=20, use_mog=False)
    print(f"Final evaluation reward (converted to Actor): {final_reward_actor:.2f}")
    
    # Save the final Actor model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(actor.state_dict(), f'models/mog_{hyperparams["num_components"]}_final_{timestamp}.pt')
    
    # Save results for reporting
    save_results_to_json(
        hyperparams["num_components"], 
        final_reward_mog, 
        training_time, 
        losses, 
        eval_rewards,
        hyperparams
    )

if __name__ == "__main__":
    main() 
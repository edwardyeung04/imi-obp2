import torch
import torch.nn as nn
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
os.makedirs('results/logprob', exist_ok=True)
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

# Create a direct regression model for pre-training
class DirectRegressionModel(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.obs_dim = np.array(env.single_observation_space.shape).prod()
        self.action_dim = np.prod(env.single_action_space.shape)
        
        # Simple architecture with batch norm for stability
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()  # Constrain outputs to [-1, 1]
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
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        normalized_actions = self.net(x)
        actions = normalized_actions * self.action_scale + self.action_bias
        return actions

# Function to copy weights from regression model to Actor
def regression_to_actor(regression_model, actor):
    # Copy common weights
    actor.fc1.weight.data.copy_(regression_model.net[0].weight.data)
    actor.fc1.bias.data.copy_(regression_model.net[0].bias.data)
    actor.fc2.weight.data.copy_(regression_model.net[3].weight.data)
    actor.fc2.bias.data.copy_(regression_model.net[3].bias.data)
    
    # Set mean output to match regression model's output
    actor.fc_mean.weight.data.copy_(regression_model.net[6].weight.data)
    actor.fc_mean.bias.data.copy_(regression_model.net[6].bias.data)
    
    # Initialize log_std with conservative values
    actor.fc_logstd.weight.data.fill_(0.01)
    actor.fc_logstd.bias.data.fill_(-1.0)  # Start with low variance
    
    return actor

# Function to evaluate the model in the environment
def evaluate_model(model, num_episodes=10, is_regression=False):
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
                
                if is_regression:
                    # Direct action prediction for regression model
                    action = model(obs_tensor)
                else:
                    # Use mean action for Actor evaluation
                    _, _, action = model.get_action(obs_tensor)
                
                action = action.cpu().squeeze().numpy()
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)

# Save results for reporting
def save_results_to_json(final_reward, training_time, losses, rewards, hyperparams=None):
    """Save training results to a JSON file for later analysis and reporting."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if hyperparams is None:
        hyperparams = {}
        
    results = {
        "method": "LogProb",
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
            "pretrain_epochs": hyperparams.get("pretrain_epochs", 20)
        }
    }
    
    # Save results
    filename = f"results/logprob/logprob_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename

# Function to train the regression model
def train_regression(model, optimizer, batch_size=4096, epochs=50, eval_interval=5):
    # Create dataset and dataloader for better batching
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
    eval_epochs = []
    best_reward = -float('inf')
    best_model_state = None
    
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    print(f"Steps per epoch: {len(train_loader)}")
    
    start_time = time.time()
    
    # MSE loss for regression
    mse_loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for obs_batch, actions_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Forward pass
            predicted_actions = model(obs_batch)
            
            # Calculate MSE loss
            loss = mse_loss_fn(predicted_actions, actions_batch)
            
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
                predicted_actions = model(obs_batch)
                val_loss = mse_loss_fn(predicted_actions, actions_batch)
                epoch_val_losses.append(val_loss.item())
        
        # Record average loss for the epoch
        train_loss = np.mean(epoch_losses)
        val_loss = np.mean(epoch_val_losses)
        losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Evaluate model in the environment
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            reward = evaluate_model(model, num_episodes=5, is_regression=True)
            eval_rewards.append(reward)
            eval_epochs.append(epoch + 1)
            print(f"Evaluation after epoch {epoch+1}: Reward = {reward:.2f}")
            
            # Save best model
            if reward > best_reward:
                best_reward = reward
                best_model_state = {
                    'model_state': model.state_dict(),
                    'reward': reward,
                    'epoch': epoch + 1
                }
                model_path = f'models/regression_best.pt'
                torch.save(best_model_state, model_path)
                print(f"New best model saved with reward {best_reward:.2f}")
    
    # Training complete
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best reward achieved: {best_reward:.2f}")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs+1), losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(eval_epochs, eval_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    
    plt.subplot(1, 3, 3)
    plt.plot(eval_epochs, eval_rewards)
    plt.title('Evaluation Rewards (Zoomed)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.ylim(max(0, min(eval_rewards) - 1000), max(eval_rewards) + 1000)  # Zoomed in view
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/regression_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Load best model for return
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state'])
    
    return model, best_reward, losses, val_losses, eval_rewards, eval_epochs, training_time

# Train the actor model using behavior cloning/log prob 
def train_actor(actor, optimizer, batch_size=4096, epochs=30, eval_interval=5):
    # Create dataset and dataloader
    dataset = TensorDataset(observations, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, pin_memory=False)
    
    losses = []
    eval_rewards = []
    best_reward = -float('inf')
    best_actor_state = None
    
    print(f"Fine-tuning for {epochs} epochs with batch size {batch_size}")
    print(f"Steps per epoch: {len(dataloader)}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        actor.train()
        epoch_losses = []
        
        for obs_batch, actions_batch in tqdm(dataloader, desc=f"Fine-tune Epoch {epoch+1}/{epochs}"):
            # Forward pass through actor
            # This will output sampled actions, log probs, and mean actions
            sampled_actions, log_probs, mean_actions = actor.get_action(obs_batch)
            
            # Compute combined loss:
            # 1. MSE loss on mean predictions for stable behavior
            mse_loss = nn.MSELoss()(mean_actions, actions_batch)
            
            # 2. Negative log probability to match the distribution
            neg_log_prob = -log_probs.mean()
            
            # Combined loss with more focus on MSE initially, gradually shifting to neg_log_prob
            # This helps avoid instability while preserving stochasticity of the policy
            progress = min(1.0, epoch / (epochs * 0.7))  # Transition over 70% of training
            log_prob_weight = 0.1 + 0.6 * progress  # Scale from 0.1 to 0.7
            mse_weight = 1.0 - log_prob_weight
            
            loss = mse_weight * mse_loss + log_prob_weight * neg_log_prob
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Record average loss for the epoch
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, MSE weight: {mse_weight:.2f}, Log Prob weight: {log_prob_weight:.2f}")
        
        # Evaluate model
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            actor.eval()
            reward = evaluate_model(actor, num_episodes=5, is_regression=False)
            eval_rewards.append(reward)
            print(f"Evaluation after epoch {epoch+1}: Reward = {reward:.2f}")
            
            # Save best model
            if reward > best_reward:
                best_reward = reward
                best_actor_state = {
                    'model_state': actor.state_dict(),
                    'reward': reward,
                    'epoch': epoch + 1
                }
                model_path = f'models/logprob_best.pt'
                torch.save(best_actor_state, model_path)
                print(f"New best model saved with reward {best_reward:.2f}")
    
    # Training complete
    training_time = time.time() - start_time
    print(f"Fine-tuning completed in {training_time:.2f} seconds")
    print(f"Best reward achieved: {best_reward:.2f}")
    
    # Load best model for final evaluation
    if best_actor_state is not None:
        actor.load_state_dict(best_actor_state['model_state'])
    
    return best_reward, losses, eval_rewards, training_time

def main():
    # Hyperparameters - only regression phase
    hyperparams = {
        "lr": 5e-5,           # Low learning rate for stability
        "batch_size": 2048,   # Smaller batch for better stability
        "epochs": 50,         # Increased epochs as requested
        "eval_interval": 5
    }
    
    # Create and train regression model
    print("===== Training Regression Model =====")
    regression_model = DirectRegressionModel(env).to(device)
    optimizer = optim.Adam(regression_model.parameters(), lr=hyperparams["lr"])
    
    # Train the regression model
    regression_model, best_reward, losses, val_losses, eval_rewards, eval_epochs, training_time = train_regression(
        regression_model,
        optimizer,
        batch_size=hyperparams["batch_size"],
        epochs=hyperparams["epochs"],
        eval_interval=hyperparams["eval_interval"]
    )
    
    # Final evaluation with more episodes
    final_reward = evaluate_model(regression_model, num_episodes=20, is_regression=True)
    print(f"Final evaluation reward: {final_reward:.2f}")
    
    # Save the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_state = {
        'model_state': regression_model.state_dict(),
        'reward': final_reward,
        'timestamp': timestamp
    }
    
    torch.save(final_model_state, f'models/regression_final_{timestamp}.pt')
    print(f"Final model saved to models/regression_final_{timestamp}.pt")
    
    # Save results for reporting
    results = {
        "method": "Regression",
        "timestamp": timestamp,
        "final_reward": float(final_reward),
        "best_reward": float(best_reward),
        "training_time_seconds": training_time,
        "loss_history": [float(x) for x in losses],
        "val_loss_history": [float(x) for x in val_losses],
        "reward_history": [float(x) for x in eval_rewards],
        "eval_epochs": eval_epochs,
        "hyperparameters": hyperparams
    }
    
    filename = f"results/logprob/regression_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main() 
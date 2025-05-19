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
from train import create_datasets, evaluate_model, device

os.makedirs('models', exist_ok=True)
os.makedirs('results/logprob', exist_ok=True)
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

class LogProbModel(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.obs_dim = np.array(env.single_observation_space.shape).prod()
        self.action_dim = np.prod(env.single_action_space.shape)
        
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()
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

def regression_to_actor(regression_model, actor):
    actor.fc1.weight.data.copy_(regression_model.net[0].weight.data)
    actor.fc1.bias.data.copy_(regression_model.net[0].bias.data)
    actor.fc2.weight.data.copy_(regression_model.net[3].weight.data)
    actor.fc2.bias.data.copy_(regression_model.net[3].bias.data)
    actor.fc_mean.weight.data.copy_(regression_model.net[6].weight.data)
    actor.fc_mean.bias.data.copy_(regression_model.net[6].bias.data)
    actor.fc_logstd.weight.data.fill_(0.01)
    actor.fc_logstd.bias.data.fill_(-1.0)
    return actor

def save_results_to_json(final_reward, training_time, losses, rewards, hyperparams=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    safe_hyperparams = {}
    if hyperparams:
        for key, value in hyperparams.items():
            if key not in ['env', 'device']:
                if hasattr(value, 'item'):
                    safe_hyperparams[key] = value.item()
                else:
                    safe_hyperparams[key] = value
    
    results = {
        "method": "LogProb",
        "timestamp": timestamp,
        "final_reward": float(final_reward),
        "training_time_seconds": training_time,
        "loss_history": [float(x) for x in losses],
        "reward_history": [float(x) for x in rewards],
        "hyperparameters": safe_hyperparams
    }
    
    filename = f"results/logprob/logprob_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename

def train_regression(model, optimizer, batch_size=4096, epochs=50, eval_interval=5):
    dataset = TensorDataset(observations, actions)
    
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
    mse_loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for obs_batch, actions_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            predicted_actions = model(obs_batch)
            loss = mse_loss_fn(predicted_actions, actions_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for obs_batch, actions_batch in val_loader:
                predicted_actions = model(obs_batch)
                val_loss = mse_loss_fn(predicted_actions, actions_batch)
                epoch_val_losses.append(val_loss.item())
        
        train_loss = np.mean(epoch_losses)
        val_loss = np.mean(epoch_val_losses)
        losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            reward = evaluate_model(model, num_episodes=5, is_regression=True)
            eval_rewards.append(reward)
            eval_epochs.append(epoch + 1)
            print(f"Evaluation after epoch {epoch+1}: Reward = {reward:.2f}")
            
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
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best reward achieved: {best_reward:.2f}")
    
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
    plt.ylim(max(0, min(eval_rewards) - 1000), max(eval_rewards) + 1000)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/regression_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state'])
    
    return model, best_reward, losses, val_losses, eval_rewards, eval_epochs, training_time

def train_logprob(model, observations, actions, **kwargs):
    batch_size = kwargs.get('batch_size', 4096)
    epochs = kwargs.get('epochs', 50)
    lr = kwargs.get('lr', 3e-4)
    eval_interval = kwargs.get('eval_interval', 5)
    env = kwargs.get('env', None)
    
    if env is None:
        from train import setup_env
        env = setup_env()
    
    train_loader, val_loader = create_datasets(observations, actions, batch_size)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    val_losses = []
    eval_rewards = []
    eval_epochs = []
    best_reward = -float('inf')
    best_model_state = None
    
    print(f"Training LogProb model for {epochs} epochs with batch size {batch_size}")
    print(f"Steps per epoch: {len(train_loader)}")
    
    mse_loss_fn = nn.MSELoss()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for obs_batch, actions_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            predicted_actions = model(obs_batch)
            loss = mse_loss_fn(predicted_actions, actions_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_batch, actions_batch in val_loader:
                predicted_actions = model(obs_batch)
                val_loss += mse_loss_fn(predicted_actions, actions_batch).item()
        
        val_loss /= len(val_loader)
        
        train_loss = np.mean(epoch_losses)
        losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            reward = evaluate_model(model, env, model_type='logprob')
            eval_rewards.append(reward)
            eval_epochs.append(epoch + 1)
            
            print(f"Evaluation after epoch {epoch+1}: Reward = {reward:.2f}")
            
            if reward > best_reward:
                best_reward = reward
                best_model_state = {
                    'model_state': model.state_dict(),
                    'reward': reward,
                    'epoch': epoch + 1
                }
                torch.save(best_model_state, f'models/logprob_best.pt')
                print(f"New best model saved with reward {reward:.2f}")
    
    training_time = time.time() - start_time
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state'])
    
    actor_model = Actor(env).to(device)
    actor_model = regression_to_actor(model, actor_model)
    
    final_reward = evaluate_model(model, env, model_type='logprob')
    print(f"Final regression model evaluation: {final_reward:.2f}")

    torch.save({
        'model_state': model.state_dict(),
        'reward': final_reward,
        'timestamp': time.strftime("%Y%m%d_%H%M%S")
    }, f'models/logprob_final.pt')
    
    return {
        'model': model,
        'actor_model': actor_model,
        'best_reward': best_reward,
        'final_reward': final_reward,
        'losses': losses,
        'val_losses': val_losses,
        'rewards': eval_rewards,
        'eval_epochs': eval_epochs,
        'training_time': training_time
    }

def main():
    hyperparams = {
        "lr": 5e-5,
        "batch_size": 2048,
        "epochs": 50,
        "eval_interval": 5
    }
    
    print("===== Training Regression Model =====")
    regression_model = LogProbModel(env).to(device)
    optimizer = optim.Adam(regression_model.parameters(), lr=hyperparams["lr"])
    
    regression_model, best_reward, losses, val_losses, eval_rewards, eval_epochs, training_time = train_regression(
        regression_model,
        optimizer,
        batch_size=hyperparams["batch_size"],
        epochs=hyperparams["epochs"],
        eval_interval=hyperparams["eval_interval"]
    )
    
    final_reward = evaluate_model(regression_model, num_episodes=20, is_regression=True)
    print(f"Final evaluation reward: {final_reward:.2f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_state = {
        'model_state': regression_model.state_dict(),
        'reward': final_reward,
        'timestamp': timestamp
    }
    
    torch.save(final_model_state, f'models/regression_final_{timestamp}.pt')
    print(f"Final model saved to models/regression_final_{timestamp}.pt")
    
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
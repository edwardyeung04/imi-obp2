# NOTE: There is a problem with the use_mixture flag; see code for details.
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
from train import create_datasets, evaluate_model, device

os.makedirs('models', exist_ok=True)
os.makedirs('results/mog', exist_ok=True)
os.makedirs('plots', exist_ok=True)

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


class MoGActor(nn.Module):
    def __init__(self, env, num_components=5):
        super().__init__()
        self.num_components = num_components
        self.obs_dim = np.array(env.single_observation_space.shape).prod()
        self.action_dim = np.prod(env.single_action_space.shape)
        

        self.fc1 = nn.Linear(self.obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        

        self.means_layer = nn.Linear(256, self.action_dim * num_components)
        self.log_stds_layer = nn.Linear(256, self.action_dim * num_components)
        self.mixture_weights_layer = nn.Linear(256, num_components)
        
  
        self.fc_mean = nn.Linear(256, self.action_dim)
        self.fc_logstd = nn.Linear(256, self.action_dim)
        

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
        
        features = self.extract_features(x)
        
        mean = self.fc_mean(features)
        log_std = self.fc_logstd(features)
        log_std = torch.tanh(log_std)
        log_std = -5 + 0.5 * (2 - (-5)) * (log_std + 1) 
        
        mog_means = self.means_layer(features)
        mog_log_stds = self.log_stds_layer(features)
        mixture_logits = self.mixture_weights_layer(features)
        
        batch_size = x.shape[0]
        mog_means = mog_means.view(batch_size, self.num_components, self.action_dim)
        mog_log_stds = mog_log_stds.view(batch_size, self.num_components, self.action_dim)
        
        mog_log_stds = torch.clamp(mog_log_stds, -5, 2)
        
        mixture_weights = F.softmax(mixture_logits, dim=-1)
        mixture_weights = torch.clamp(mixture_weights, min=1e-6, max=1.0)
        mixture_weights = mixture_weights / mixture_weights.sum(dim=-1, keepdim=True)
        
        return mean, log_std, mog_means, mog_log_stds, mixture_weights
    
    def sample_from_mixture(self, mog_means, mog_log_stds, mixture_weights, temperature=1.0):
        batch_size = mog_means.shape[0]
        
        if temperature != 1.0:
            log_weights = torch.log(mixture_weights) / temperature
            mixture_weights = F.softmax(log_weights, dim=-1)
        
        component_indices = torch.multinomial(mixture_weights, 1).squeeze(-1)
        
        batch_indices = torch.arange(batch_size, device=mog_means.device)
        selected_means = mog_means[batch_indices, component_indices]
        selected_log_stds = mog_log_stds[batch_indices, component_indices]
        
        mog_std = torch.exp(selected_log_stds)
        mog_normal = torch.distributions.Normal(selected_means, mog_std)
        mog_x_t = mog_normal.rsample()
        mog_y_t = torch.tanh(mog_x_t)
        mog_action = mog_y_t * self.action_scale + self.action_bias
        
        return mog_action, selected_means
    
    def get_action(self, x, use_mixture=True, temperature=1.0):
        mean, log_std, mog_means, mog_log_stds, mixture_weights = self(x)
        batch_size = x.shape[0]
        
        if not self.training and use_mixture:
            mog_action, selected_means = self.sample_from_mixture(
                mog_means, mog_log_stds, mixture_weights, temperature
            )
            return mog_action, None, selected_means
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        bc_action = y_t * self.action_scale + self.action_bias
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        if self.training:
            if torch.isnan(mixture_weights).any() or torch.isinf(mixture_weights).any() or (mixture_weights < 0).any():
                mixture_weights = torch.ones_like(mixture_weights) / self.num_components
            
            component_indices = torch.multinomial(mixture_weights, 1).squeeze(-1)
            
            batch_indices = torch.arange(batch_size, device=x.device)
            selected_means = mog_means[batch_indices, component_indices]
            selected_log_stds = mog_log_stds[batch_indices, component_indices]
            
            mog_std = torch.exp(selected_log_stds)
            mog_normal = torch.distributions.Normal(selected_means, mog_std)
            mog_x_t = mog_normal.rsample()
            mog_y_t = torch.tanh(mog_x_t)
            mog_action = mog_y_t * self.action_scale + self.action_bias
            
            component_log_probs = []
            for c in range(self.num_components):
                std_c = torch.exp(mog_log_stds[:, c])
                normal_c = torch.distributions.Normal(mog_means[:, c], std_c)
                log_prob_c = normal_c.log_prob(mog_x_t)
                log_prob_c -= torch.log(self.action_scale * (1 - mog_y_t.pow(2)) + 1e-6)
                log_prob_c = log_prob_c.sum(1, keepdim=True)
                component_log_probs.append(log_prob_c)
            
            component_log_probs = torch.stack(component_log_probs, dim=-1)
            safe_mixture_weights = torch.clamp(mixture_weights, min=1e-8)
            log_probs_weighted = component_log_probs + torch.log(safe_mixture_weights).unsqueeze(1)
            
            mog_log_prob = torch.logsumexp(log_probs_weighted, dim=-1)
            
            if torch.isnan(mog_log_prob).any() or torch.isinf(mog_log_prob).any():
                mog_log_prob = torch.ones_like(mog_log_prob) * -10.0
            
            return bc_action, [log_prob, mog_log_prob], mean
        else:
            mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
            return mean_action, log_prob, mean_action

def mog_to_actor(mog_model, actor_model):
    mog_state_dict = mog_model.state_dict()
    actor_state_dict = actor_model.state_dict()
    
    for key in actor_state_dict.keys():
        if key in mog_state_dict:
            actor_state_dict[key] = mog_state_dict[key]
    
    actor_model.load_state_dict(actor_state_dict)
    return actor_model

def evaluate_model(model, env, num_episodes=10, model_type='mog', use_mixture=True):

    if not use_mixture:
        actor = Actor(env).to(device)
        actor = mog_to_actor(model, actor)
    else:
        actor = model
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not (done or truncated):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                _, _, action = actor.get_action(obs_tensor)
                action = action.cpu().squeeze().numpy()
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)

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
    
    filename = f"results/mog/mog_{num_components}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename

def compute_bc_loss(predicted_actions, expert_actions):
    return F.mse_loss(predicted_actions, expert_actions)

def compute_mog_loss(log_probs):
    return -log_probs[1].mean()

def train_mog(model, observations, actions, **kwargs):
    batch_size = kwargs.get('batch_size', 2048)
    epochs = kwargs.get('epochs', 50)
    lr = kwargs.get('lr', 3e-4)
    eval_interval = kwargs.get('eval_interval', 5)
    env = kwargs.get('env', None)
    num_components = kwargs.get('num_components', 5)
    bc_weight = kwargs.get('bc_weight', 0.5)
    use_mixture = kwargs.get('use_mixture', True)
    
    if env is None:
        from train import setup_env
        env = setup_env()
    
    if not isinstance(model, MoGActor):
        model = MoGActor(env, num_components=num_components).to(device)
    
    train_loader, val_loader = create_datasets(observations, actions, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    val_losses = []
    eval_rewards = []
    eval_rewards_mixture = []
    eval_epochs = []
    best_reward = -float('inf')
    best_model_state = None
    
    print(f"Training MoG model with {num_components} components for {epochs} epochs")
    print(f"Batch size: {batch_size}, BC weight: {bc_weight}")
    print(f"Steps per epoch: {len(train_loader)}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for obs_batch, actions_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            actions_pred, log_probs, _ = model.get_action(obs_batch)
            
            bc_loss = compute_bc_loss(actions_pred, actions_batch)
            mog_loss = compute_mog_loss(log_probs)
            
            loss = bc_weight * bc_loss + (1 - bc_weight) * mog_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_batch, actions_batch in val_loader:
                actions_pred, _, _ = model.get_action(obs_batch)
                val_loss += compute_bc_loss(actions_pred, actions_batch).item()
        
        val_loss /= len(val_loader)
        
        train_loss = np.mean(epoch_losses)
        losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            reward_mean = evaluate_model(model, env, num_episodes=10, model_type='mog', use_mixture=False)
            reward_mixture = evaluate_model(model, env, num_episodes=10, model_type='mog', use_mixture=True)
            reward = max(reward_mean, reward_mixture)
            eval_rewards.append(reward_mean)
            eval_rewards_mixture.append(reward_mixture)
            eval_epochs.append(epoch + 1)
            print(f"Evaluation after epoch {epoch+1}:")
            print(f"  Mean only: Reward = {reward_mean:.2f}")
            print(f"  Mixture sampling: Reward = {reward_mixture:.2f}")
            if reward > best_reward:
                best_reward = reward
                best_model_state = {
                    'model_state': model.state_dict(),
                    'reward': reward,
                    'reward_mean': reward_mean,
                    'reward_mixture': reward_mixture,
                    'epoch': epoch + 1,
                    'num_components': num_components
                }
                torch.save(best_model_state, f'models/mog_{num_components}_best.pt')
                print(f"New best model saved with reward {reward:.2f}")
    
    training_time = time.time() - start_time
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state'])
    
    final_reward_mean = evaluate_model(model, env, num_episodes=10, model_type='mog', use_mixture=False)
    final_reward_mixture = evaluate_model(model, env, num_episodes=10, model_type='mog', use_mixture=True)
    final_reward = max(final_reward_mean, final_reward_mixture)
    
    print(f"Final evaluation rewards:")
    print(f"  Mean only: {final_reward_mean:.2f}")
    print(f"  Mixture sampling: {final_reward_mixture:.2f}")
    
    torch.save({
        'model_state': model.state_dict(),
        'reward_mean': final_reward_mean,
        'reward_mixture': final_reward_mixture,
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
        'num_components': num_components
    }, f'models/mog_{num_components}_final.pt')
    
    actor = Actor(env).to(device)
    actor = mog_to_actor(model, actor)
    actor_path = f'models/mog_{num_components}_actor_converted.pt'
    torch.save(actor.state_dict(), actor_path)
    
    return {
        'model': model,
        'best_reward': best_reward,
        'final_reward': final_reward,
        'losses': losses,
        'val_losses': val_losses,
        'rewards': eval_rewards,
        'rewards_mixture': eval_rewards_mixture,
        'eval_epochs': eval_epochs,
        'training_time': training_time
    }

def main():
    hyperparams = {
        "lr": 3e-4,
        "batch_size": 4096,
        "epochs": 50,
        "eval_interval": 5,
        "num_components": 5,
        "bc_weight": 0.5
    }
    
    print("===== Training Mixture of Gaussians Model =====")
    model = MoGActor(env=gym.make("HalfCheetah-v4"), num_components=hyperparams["num_components"]).to(device)
    
    observations, actions = load_data()
    
    results = train_mog(
        model,
        observations,
        actions,
        batch_size=hyperparams["batch_size"],
        epochs=hyperparams["epochs"],
        lr=hyperparams["lr"],
        eval_interval=hyperparams["eval_interval"],
        num_components=hyperparams["num_components"],
        bc_weight=hyperparams["bc_weight"]
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        "method": f"MoG-{hyperparams['num_components']}",
        "timestamp": timestamp,
        "final_reward_mean": float(results["rewards"][-1]),
        "final_reward_mixture": float(results["rewards_mixture"][-1]),
        "best_reward": float(results["best_reward"]),
        "training_time_seconds": results["training_time"],
        "loss_history": [float(x) for x in results["losses"]],
        "val_loss_history": [float(x) for x in results["val_losses"]],
        "reward_history_mean": [float(x) for x in results["rewards"]],
        "reward_history_mixture": [float(x) for x in results["rewards_mixture"]],
        "eval_epochs": results["eval_epochs"],
        "hyperparameters": hyperparams
    }
    
    filename = f"results/mog/mog_{hyperparams['num_components']}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Results saved to {filename}")

def load_data():
    print("Loading data...")
    data = torch.load('data/halfcheetah_v4_data.pt', map_location=device, weights_only=False)
    observations = data['observations'].to(device)
    actions = data['actions'].to(device)
    print(f"Loaded {len(observations)} observation-action pairs")
    print(f"Mean reward in dataset: {data['mean_reward']}, Std: {data['std_reward']}")
    return observations, actions

if __name__ == "__main__":
    main() 
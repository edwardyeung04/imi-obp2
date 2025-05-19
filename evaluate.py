import torch
import numpy as np
import gymnasium as gym
from actor_impl import Actor
import os
import json
from datetime import datetime

def evaluate_model(model_path, num_episodes=20, render=False, save_results=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = gym.make("HalfCheetah-v4", render_mode="human" if render else None)
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    
    actor = Actor(env).to(device)
    print(f"Loading model from {model_path}")
    
    model_data = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(model_data, dict) and 'model_state' in model_data:
        actor.load_state_dict(model_data['model_state'])
        model_info = {k: v for k, v in model_data.items() if k != 'model_state'}
        print(f"Model info: {model_info}")
    else:
        actor.load_state_dict(model_data)
    
    actor.eval()
    
    rewards = []
    for episode in range(num_episodes):
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
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    print("\nEvaluation Results:")
    print(f"Number of episodes: {num_episodes}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Standard deviation: {std_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    
    if save_results:
        os.makedirs('results/evaluations', exist_ok=True)
        model_name = os.path.basename(model_path).split('.')[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "model": model_name,
            "model_path": model_path,
            "timestamp": timestamp,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "min_reward": float(min_reward),
            "max_reward": float(max_reward),
            "rewards": [float(r) for r in rewards],
            "num_episodes": num_episodes
        }
        
        result_path = f"results/evaluations/{model_name}_eval_{timestamp}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {result_path}")
    
    return mean_reward, std_reward

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", type=str, default="models/logprob_best.pt", 
                      help="Path to the model file")
    parser.add_argument("--episodes", type=int, default=20,
                      help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
                      help="Render the environment")
    parser.add_argument("--no-save", action="store_true",
                      help="Don't save evaluation results")
    
    args = parser.parse_args()
    evaluate_model(args.model, args.episodes, args.render, not args.no_save) 
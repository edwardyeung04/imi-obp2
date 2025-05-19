import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from datetime import datetime

def load_results(directory):
    results = []
    for method_dir in os.listdir(directory):
        subdir = os.path.join(directory, method_dir)
        if not os.path.isdir(subdir):
            continue
        method_files = glob.glob(os.path.join(subdir, "*.json"))
        if method_files:
            method_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_file = method_files[0]
            with open(latest_file, 'r') as f:
                data = json.load(f)
                data['source_file'] = os.path.basename(latest_file)
                results.append(data)
    expert_eval_dir = os.path.join(directory, 'evaluations')
    if os.path.isdir(expert_eval_dir):
        expert_files = glob.glob(os.path.join(expert_eval_dir, '*expert*.json'))
        if not expert_files:
            expert_files = glob.glob(os.path.join(expert_eval_dir, '*.json'))
        if expert_files:
            expert_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            expert_file = expert_files[0]
            with open(expert_file, 'r') as f:
                data = json.load(f)
                if 'mean_reward' in data:
                    expert_result = {
                        'method': 'expert',
                        'final_reward': data['mean_reward'],
                        'std_reward': data.get('std_reward'),
                        'min_reward': data.get('min_reward'),
                        'max_reward': data.get('max_reward'),
                        'source_file': os.path.basename(expert_file)
                    }
                    results.append(expert_result)
    return results

def plot_rewards_comparison(results, save_path=None):
    filtered = [r for r in results if isinstance(r.get('method'), str) and r.get('method').strip()]
    if len(filtered) < len(results):
        print("Warning: Some results were skipped due to missing or invalid 'method' fields.")
        for r in results:
            if not (isinstance(r.get('method'), str) and r.get('method').strip()):
                print("  Skipped result:", r.get('source_file', 'unknown'), "method:", r.get('method'))
    methods = [r.get('method') for r in filtered]
    rewards = []
    for r in filtered:
        if 'final_reward' in r:
            rewards.append(r['final_reward'])
        elif 'best_reward' in r:
            rewards.append(r['best_reward'])
        else:
            rewards.append(0)
    sorted_indices = np.argsort(rewards)[::-1]
    sorted_methods = [methods[i] for i in sorted_indices]
    sorted_rewards = [rewards[i] for i in sorted_indices]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_methods, sorted_rewards, color=['blue', 'green', 'red', 'purple'][:len(sorted_methods)])
    for bar, value in zip(bars, sorted_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, value + 100, f"{value:.1f}", ha='center', va='bottom', fontweight='bold')
    plt.title('Comparison of Final Rewards Across Methods', fontsize=16)
    plt.ylabel('Reward', fontsize=14)
    plt.xlabel('Method', fontsize=14)
    plt.ylim(0, max(sorted_rewards) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Reward comparison plot saved to {save_path}")
    plt.show()

def plot_training_curves(results, metric='reward', save_path=None):
    plt.figure(figsize=(12, 8))
    colors = {
        'logprob': 'blue',
        'mog': 'green',
        'diffusion': 'red',
        'autoreg': 'purple',
        'AutoregressiveDiscretized': 'purple'
    }
    if metric == 'reward':
        plt.title('Evaluation Rewards During Training', fontsize=16)
        plt.ylabel('Average Reward', fontsize=14)
        y_label = 'reward_history'
        alt_label = 'reward_history_sampling'
    elif metric == 'loss':
        plt.title('Training Loss Curves', fontsize=16)
        plt.ylabel('Loss', fontsize=14)
        y_label = 'loss_history'
        alt_label = 'loss_history'
    plt.xlabel('Epoch', fontsize=14)
    plt.grid(True, alpha=0.3)
    for result in results:
        method = result.get('method')
        if method == 'expert':
            continue
        if method in colors:
            color = colors[method]
        else:
            matched = False
            for key in colors:
                if isinstance(method, str) and method.startswith(key):
                    color = colors[key]
                    matched = True
                    break
            if not matched:
                color = 'gray'
        if y_label in result:
            y_data = result[y_label]
        elif alt_label in result:
            y_data = result[alt_label]
        else:
            print(f"Skipping {method}: no {y_label} or {alt_label} found.")
            continue
        if metric == 'loss':
            x_data = list(range(1, len(y_data) + 1))
        elif metric == 'reward' and 'eval_epochs' in result:
            x_data = result['eval_epochs']
        else:
            x_data = list(range(1, len(y_data) + 1))
        plt.plot(x_data, y_data, '-o', label=method, color=color, linewidth=2, markersize=6)
    plt.legend(fontsize=12)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"{metric.capitalize()} curves plot saved to {save_path}")
    plt.show()

def create_summary_table(results, save_path=None):
    data = []
    for result in results:
        method = result.get('method')
        final_reward = result.get('final_reward', result.get('best_reward', 0))
        training_time = result.get('training_time_seconds', 0)
        training_time_min = training_time / 60
        epochs = result.get('hyperparameters', {}).get('epochs', 0)
        if method == 'expert':
            std = result.get('std_reward', None)
            if std is not None:
                reward_note = f"{final_reward:.1f} (std: {std:.1f})"
            else:
                reward_note = f"{final_reward:.1f}"
        elif method == 'AutoregressiveDiscretized' or (isinstance(method, str) and method.startswith('autoreg')):
            sampling_reward = result.get('final_reward_sampling', final_reward)
            greedy_reward = result.get('final_reward_greedy', final_reward)
            reward_note = f"{final_reward:.1f} (S:{sampling_reward:.1f}, G:{greedy_reward:.1f})"
        else:
            reward_note = f"{final_reward:.1f}"
        data.append({
            'Method': method,
            'Reward': final_reward,
            'Reward (formatted)': reward_note,
            'Training Time (min)': training_time_min,
            'Epochs': epochs,
            'Source File': result.get('source_file', 'unknown')
        })
    df = pd.DataFrame(data)
    df = df.sort_values('Reward', ascending=False)
    display_df = df.copy()
    display_df['Training Time (min)'] = display_df['Training Time (min)'].map('{:.1f}'.format)
    print("\n===== Method Comparison Summary =====")
    summary_table = display_df[['Method', 'Reward (formatted)', 'Training Time (min)', 'Epochs', 'Source File']]
    print(summary_table.to_string(index=False))
    if save_path:
        display_df.to_csv(save_path, index=False)
        print(f"Summary table saved to {save_path}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Compare imitation learning methods for HalfCheetah")
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory containing result files (default: results)')
    parser.add_argument('--metric', type=str, choices=['reward', 'loss'], default='reward',
                      help='Which metric to plot for training curves (default: reward)')
    parser.add_argument('--output_dir', type=str, default='comparison',
                      help='Directory to save comparison outputs (default: comparison)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found! Make sure you've run the training methods first.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    reward_plot_path = os.path.join(args.output_dir, f"reward_comparison_{timestamp}.png")
    plot_rewards_comparison(results, save_path=reward_plot_path)
    
    curves_plot_path = os.path.join(args.output_dir, f"{args.metric}_curves_{timestamp}.png")
    plot_training_curves(results, metric=args.metric, save_path=curves_plot_path)
    
    table_path = os.path.join(args.output_dir, f"method_summary_{timestamp}.csv")
    create_summary_table(results, save_path=table_path)
    
    print(f"\nAll comparison outputs saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 
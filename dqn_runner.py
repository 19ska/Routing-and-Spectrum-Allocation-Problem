import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from rsaenv import RSAEnv


class TrainingMetricsCallback(BaseCallback):
    # Track episode rewards and blocking rates during training
    def __init__(self, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_blocking_rates = []
        self.current_episode_reward = 0
        self.episode_count = 0

    def _on_step(self):
        self.current_episode_reward += self.locals['rewards'][0]

        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)

            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                blocking_rate = self.locals['infos'][0].get('blocking_rate', 0.0)
                self.episode_blocking_rates.append(blocking_rate)
            else:
                self.episode_blocking_rates.append(0.0)

            self.episode_count += 1
            self.current_episode_reward = 0

            if self.verbose > 0 and self.episode_count % 100 == 0:
                print(f"Episode {self.episode_count}: "
                      f"Avg Reward (last 10) = {np.mean(self.episode_rewards[-10:]):.2f}, "
                      f"Avg Blocking Rate (last 10) = {np.mean(self.episode_blocking_rates[-10:]):.4f}")

        return True


class MultiFileEnvWrapper(gym.Wrapper):
    # Cycles through multiple request files during training
    def __init__(self, request_files, capacity=20):
        self.request_files = request_files
        self.capacity = capacity
        self.current_file_idx = 0
        env = RSAEnv(request_file=self.request_files[0], capacity=capacity)
        super().__init__(env)

    def reset(self, **kwargs):
        request_file = self.request_files[self.current_file_idx]
        self.current_file_idx = (self.current_file_idx + 1) % len(self.request_files)
        obs, info = self.env.reset(options={'request_file': request_file})
        return obs, info


def train_dqn_agent(capacity=20, total_timesteps=1000000, model_name='dqn_rsa'):
    # Train a DQN agent for the RSA problem
    print(f"Training DQN agent with capacity={capacity}")

    train_files = sorted(glob.glob('data/train/requests-*.csv'))
    print(f"Found {len(train_files)} training files")

    env = MultiFileEnvWrapper(train_files, capacity=capacity)
    callback = TrainingMetricsCallback(verbose=1)

    # Tuned hyperparameters
    learning_rate = 1e-4
    buffer_size = 50000
    learning_starts = 1000
    batch_size = 64
    tau = 1.0
    gamma = 0.99
    target_update_interval = 1000
    exploration_fraction = 0.3
    exploration_initial_eps = 1.0
    exploration_final_eps = 0.05

    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        verbose=1
    )

    print("Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(model_name)
    print(f"Model saved to {model_name}.zip")

    env.close()
    return model, callback


def evaluate_agent(model_path, capacity=20, eval_episodes=1000):
    # Evaluate a trained DQN agent on the evaluation dataset
    print(f"Evaluating agent from {model_path}")

    eval_files = sorted(glob.glob('data/eval/requests-*.csv'))[:eval_episodes]
    print(f"Evaluating on {len(eval_files)} files")

    model = DQN.load(model_path)

    episode_rewards = []
    episode_blocking_rates = []

    for i, request_file in enumerate(eval_files):
        env = RSAEnv(request_file=request_file, capacity=capacity)
        obs, _ = env.reset()

        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_blocking_rates.append(info.get('blocking_rate', 0.0))

        if (i + 1) % 100 == 0:
            print(f"Evaluated {i + 1}/{len(eval_files)} episodes")

        env.close()

    return episode_rewards, episode_blocking_rates


def plot_training_results(callback, capacity, save_prefix='training'):
    # Plot training results: learning curve and objective function
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    episodes = np.arange(len(callback.episode_rewards))

    def rolling_avg(data, window=10):
        avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            avg.append(np.mean(data[start_idx:i + 1]))
        return avg

    avg_rewards = rolling_avg(callback.episode_rewards)
    avg_blocking_rates = rolling_avg(callback.episode_blocking_rates)

    # Learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, avg_rewards, linewidth=1.5)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Episode Reward (last 10 episodes)', fontsize=12)
    plt.title(f'Learning Curve (Capacity={capacity})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'{save_prefix}_capacity{capacity}_learning_curve.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved {plot_path}")

    # Objective function
    objective = [1 - br for br in avg_blocking_rates]
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, objective, linewidth=1.5, color='green')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Objective (1 - Blocking Rate)', fontsize=12)
    plt.title(f'Objective Function over Training (Capacity={capacity})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'{save_prefix}_capacity{capacity}_objective.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved {plot_path}")


def plot_evaluation_results(rewards, blocking_rates, capacity, save_prefix='eval'):
    # Plot evaluation results
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    episodes = np.arange(len(rewards))

    def rolling_avg(data, window=10):
        avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            avg.append(np.mean(data[start_idx:i + 1]))
        return avg

    avg_blocking_rates = rolling_avg(blocking_rates)

    objective = [1 - br for br in avg_blocking_rates]
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, objective, linewidth=1.5, color='blue')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Objective (1 - Blocking Rate)', fontsize=12)
    plt.title(f'Evaluation Performance (Capacity={capacity})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'{save_prefix}_capacity{capacity}_objective.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved {plot_path}")


def save_results_summary(callback_20, eval_rewards_20, eval_blocking_20, 
                        callback_10, eval_rewards_10, eval_blocking_10):
    # Save comprehensive results to text file for README population
    with open('results.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RSA DQN TRAINING RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        # Training configuration
        f.write("TRAINING CONFIGURATION:\n")
        f.write("- Total timesteps per model: 1,000,000\n")
        f.write("- Training files: 10,000\n")
        f.write("- Evaluation files: 1,000\n")
        f.write("- Episodes per model: ~10,000\n\n")
        
        # Capacity=20 Results
        f.write("CAPACITY = 20 RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        # Training metrics
        final_reward_20 = np.mean(callback_20.episode_rewards[-10:]) if len(callback_20.episode_rewards) >= 10 else np.mean(callback_20.episode_rewards)
        final_training_blocking_20 = np.mean(callback_20.episode_blocking_rates[-10:]) if len(callback_20.episode_blocking_rates) >= 10 else np.mean(callback_20.episode_blocking_rates)
        initial_reward_20 = np.mean(callback_20.episode_rewards[:10]) if len(callback_20.episode_rewards) >= 10 else callback_20.episode_rewards[0]
        
        f.write(f"Training Episodes: {len(callback_20.episode_rewards)}\n")
        f.write(f"Initial Training Reward (first 10 episodes): {initial_reward_20:.2f}\n")
        f.write(f"Final Training Reward (last 10 episodes): {final_reward_20:.2f}\n")
        f.write(f"Final Training Blocking Rate: {final_training_blocking_20:.4f}\n")
        f.write(f"Final Training Objective: {1 - final_training_blocking_20:.4f}\n")
        
        # Evaluation metrics
        avg_eval_reward_20 = np.mean(eval_rewards_20)
        avg_eval_blocking_20 = np.mean(eval_blocking_20)
        final_eval_blocking_20 = np.mean(eval_blocking_20[-10:])
        
        f.write(f"Average Evaluation Reward: {avg_eval_reward_20:.2f}\n")
        f.write(f"Average Evaluation Blocking Rate: {avg_eval_blocking_20:.4f}\n")
        f.write(f"Final Evaluation Blocking Rate (last 10): {final_eval_blocking_20:.4f}\n")
        f.write(f"Final Evaluation Objective: {1 - final_eval_blocking_20:.4f}\n\n")
        
        # Capacity=10 Results
        f.write("CAPACITY = 10 RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        # Training metrics
        final_reward_10 = np.mean(callback_10.episode_rewards[-10:]) if len(callback_10.episode_rewards) >= 10 else np.mean(callback_10.episode_rewards)
        final_training_blocking_10 = np.mean(callback_10.episode_blocking_rates[-10:]) if len(callback_10.episode_blocking_rates) >= 10 else np.mean(callback_10.episode_blocking_rates)
        initial_reward_10 = np.mean(callback_10.episode_rewards[:10]) if len(callback_10.episode_rewards) >= 10 else callback_10.episode_rewards[0]
        
        f.write(f"Training Episodes: {len(callback_10.episode_rewards)}\n")
        f.write(f"Initial Training Reward (first 10 episodes): {initial_reward_10:.2f}\n")
        f.write(f"Final Training Reward (last 10 episodes): {final_reward_10:.2f}\n")
        f.write(f"Final Training Blocking Rate: {final_training_blocking_10:.4f}\n")
        f.write(f"Final Training Objective: {1 - final_training_blocking_10:.4f}\n")
        
        # Evaluation metrics
        avg_eval_reward_10 = np.mean(eval_rewards_10)
        avg_eval_blocking_10 = np.mean(eval_blocking_10)
        final_eval_blocking_10 = np.mean(eval_blocking_10[-10:])
        
        f.write(f"Average Evaluation Reward: {avg_eval_reward_10:.2f}\n")
        f.write(f"Average Evaluation Blocking Rate: {avg_eval_blocking_10:.4f}\n")
        f.write(f"Final Evaluation Blocking Rate (last 10): {final_eval_blocking_10:.4f}\n")
        f.write(f"Final Evaluation Objective: {1 - final_eval_blocking_10:.4f}\n\n")
        
        # Comparison table
        f.write("COMPARISON TABLE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Metric':<35} {'Capacity=20':<15} {'Capacity=10':<15}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Final Training Reward':<35} {final_reward_20:<15.2f} {final_reward_10:<15.2f}\n")
        f.write(f"{'Final Training Objective':<35} {1-final_training_blocking_20:<15.4f} {1-final_training_blocking_10:<15.4f}\n")
        f.write(f"{'Evaluation Blocking Rate':<35} {final_eval_blocking_20:<15.4f} {final_eval_blocking_10:<15.4f}\n")
        f.write(f"{'Evaluation Objective':<35} {1-final_eval_blocking_20:<15.4f} {1-final_eval_blocking_10:<15.4f}\n")
        f.write(f"{'Training Episodes':<35} {len(callback_20.episode_rewards):<15} {len(callback_10.episode_rewards):<15}\n\n")
        
        # Key milestones
        f.write("KEY LEARNING MILESTONES:\n")
        f.write("-" * 30 + "\n")
        
        # Find when rewards became positive for capacity=20
        positive_episode_20 = next((i for i, r in enumerate(callback_20.episode_rewards) if r > 0), len(callback_20.episode_rewards))
        f.write(f"Capacity=20 - First positive reward at episode: {positive_episode_20}\n")
        
        # Find when rewards became positive for capacity=10
        positive_episode_10 = next((i for i, r in enumerate(callback_10.episode_rewards) if r > 0), len(callback_10.episode_rewards))
        f.write(f"Capacity=10 - First positive reward at episode: {positive_episode_10}\n\n")
        
        # Performance percentiles
        f.write("EVALUATION PERFORMANCE DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        f.write("Capacity=20 Blocking Rate Percentiles:\n")
        f.write(f"  Min: {np.min(eval_blocking_20):.4f}\n")
        f.write(f"  25th: {np.percentile(eval_blocking_20, 25):.4f}\n")
        f.write(f"  50th (Median): {np.percentile(eval_blocking_20, 50):.4f}\n")
        f.write(f"  75th: {np.percentile(eval_blocking_20, 75):.4f}\n")
        f.write(f"  Max: {np.max(eval_blocking_20):.4f}\n\n")
        
        f.write("Capacity=10 Blocking Rate Percentiles:\n")
        f.write(f"  Min: {np.min(eval_blocking_10):.4f}\n")
        f.write(f"  25th: {np.percentile(eval_blocking_10, 25):.4f}\n")
        f.write(f"  50th (Median): {np.percentile(eval_blocking_10, 50):.4f}\n")
        f.write(f"  75th: {np.percentile(eval_blocking_10, 75):.4f}\n")
        f.write(f"  Max: {np.max(eval_blocking_10):.4f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Results saved for README population\n")
        f.write("=" * 60 + "\n")
    
    print("Results summary saved to results.txt")


def main():
    # Main training and evaluation pipeline
    # Train capacity=20
    print("=" * 50)
    print("PART 1: Training with Capacity=20")
    print("=" * 50)
    model_20, callback_20 = train_dqn_agent(
        capacity=20,
        total_timesteps=1000000,
        model_name='dqn_rsa_capacity20'
    )
    plot_training_results(callback_20, capacity=20, save_prefix='training')

    print("\nEvaluating on eval dataset (capacity=20)...")
    eval_rewards_20, eval_blocking_20 = evaluate_agent(
        'dqn_rsa_capacity20',
        capacity=20,
        eval_episodes=1000
    )
    plot_evaluation_results(eval_rewards_20, eval_blocking_20, capacity=20, save_prefix='eval')

    # Train capacity=10
    print("\n" + "=" * 50)
    print("PART 2: Training with Capacity=10")
    print("=" * 50)
    model_10, callback_10 = train_dqn_agent(
        capacity=10,
        total_timesteps=1000000,
        model_name='dqn_rsa_capacity10'
    )
    plot_training_results(callback_10, capacity=10, save_prefix='training')

    print("\nEvaluating on eval dataset (capacity=10)...")
    eval_rewards_10, eval_blocking_10 = evaluate_agent(
        'dqn_rsa_capacity10',
        capacity=10,
        eval_episodes=1000
    )
    plot_evaluation_results(eval_rewards_10, eval_blocking_10, capacity=10, save_prefix='eval')

    # Save comprehensive results
    save_results_summary(callback_20, eval_rewards_20, eval_blocking_20,
                        callback_10, eval_rewards_10, eval_blocking_10)

    print("\n" + "=" * 50)
    print("Training and Evaluation Complete!")
    print("=" * 50)
    print(f"Capacity=20 - Final Avg Blocking Rate (Eval): {np.mean(eval_blocking_20[-10:]):.4f}")
    print(f"Capacity=10 - Final Avg Blocking Rate (Eval): {np.mean(eval_blocking_10[-10:]):.4f}")
    print("\nDetailed results saved to results.txt")


if __name__ == '__main__':
    main()

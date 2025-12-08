"""Quick training test with minimal timesteps to verify DQN works"""

import glob
from stable_baselines3 import DQN
from dqn_runner import MultiFileEnvWrapper, TrainingMetricsCallback

print("Testing DQN training with minimal timesteps...")

# Load a few training files
train_files = sorted(glob.glob('data/train/requests-*.csv'))[:20]
print(f"Using {len(train_files)} training files")

# Create environment wrapper
env = MultiFileEnvWrapper(train_files, capacity=20)

# Create callback
callback = TrainingMetricsCallback(verbose=1)

# Create DQN model with minimal settings
print("Creating DQN model...")
model = DQN(
    'MlpPolicy',
    env,
    learning_rate=1e-4,
    buffer_size=10000,
    learning_starts=100,
    batch_size=64,
    verbose=1
)

# Train for just 5000 timesteps to verify it works
print("Training for 5000 timesteps...")
model.learn(total_timesteps=5000, callback=callback)

print(f"\nCompleted {callback.episode_count} episodes")
print(f"Average reward (last 10 eps): {sum(callback.episode_rewards[-10:]) / min(10, len(callback.episode_rewards)):.2f}")
print(f"Average blocking rate (last 10 eps): {sum(callback.episode_blocking_rates[-10:]) / min(10, len(callback.episode_blocking_rates)):.4f}")

# Save model
model.save("test_dqn_model")
print("\nTest model saved to test_dqn_model.zip")

env.close()
print("\nQuick training test PASSED! Ready for full training.")

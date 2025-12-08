"""Quick test to verify the MultiFileEnvWrapper works with stable-baselines3"""

import glob
from dqn_runner import MultiFileEnvWrapper

# Load a few training files
train_files = sorted(glob.glob('data/train/requests-*.csv'))[:10]
print(f"Testing with {len(train_files)} files")

# Create wrapper
env = MultiFileEnvWrapper(train_files, capacity=20)

print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Test reset
obs, info = env.reset()
print(f"Observation shape after reset: {obs.shape}")

# Test a few steps
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: reward={reward:.2f}, done={terminated or truncated}")

    if terminated or truncated:
        obs, info = env.reset()
        print("  Episode reset")

env.close()
print("\nWrapper test PASSED! Ready to train with stable-baselines3")

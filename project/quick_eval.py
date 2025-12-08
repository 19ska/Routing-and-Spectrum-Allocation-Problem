"""
Quick evaluation script to test a trained model on a single file.
Useful for debugging and quick performance checks.
"""

import sys
from stable_baselines3 import DQN
from rsaenv import RSAEnv


def evaluate_single_file(model_path, request_file, capacity=20):
    """
    Evaluate a trained model on a single request file.

    Args:
        model_path: Path to the trained model (without .zip extension)
        request_file: Path to the request CSV file
        capacity: Link capacity

    Returns:
        Episode reward, blocking rate, and detailed stats
    """
    print(f"Loading model from {model_path}.zip")
    model = DQN.load(model_path)

    print(f"Evaluating on {request_file}")
    env = RSAEnv(request_file=request_file, capacity=capacity)
    obs, _ = env.reset()

    episode_reward = 0
    steps = 0
    blocked = 0
    allocated = 0
    invalid = 0

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        steps += 1

        if info.get('blocked', False):
            blocked += 1
        if info.get('allocated', False):
            allocated += 1
        if info.get('invalid_action', False):
            invalid += 1

        done = terminated or truncated

    blocking_rate = info.get('blocking_rate', 0.0)

    print(f"\nResults:")
    print(f"  Total requests: {steps}")
    print(f"  Allocated: {allocated}")
    print(f"  Blocked: {blocked}")
    print(f"  Invalid actions: {invalid}")
    print(f"  Episode reward: {episode_reward:.2f}")
    print(f"  Blocking rate: {blocking_rate:.4f}")
    print(f"  Objective (1 - blocking rate): {1 - blocking_rate:.4f}")

    env.close()

    return episode_reward, blocking_rate


def main():
    """Main function with command-line interface."""
    if len(sys.argv) < 3:
        print("Usage: python3 quick_eval.py <model_path> <request_file> [capacity]")
        print("\nExamples:")
        print("  python3 quick_eval.py dqn_rsa_capacity20 data/eval/requests-0.csv 20")
        print("  python3 quick_eval.py dqn_rsa_capacity10 data/train/requests-18.csv 10")
        sys.exit(1)

    model_path = sys.argv[1]
    request_file = sys.argv[2]
    capacity = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    evaluate_single_file(model_path, request_file, capacity)


if __name__ == '__main__':
    main()

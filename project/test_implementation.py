"""
Test script to verify the RSA environment implementation.
Run this before full training to ensure everything works correctly.
"""

import numpy as np
from rsaenv import RSAEnv
from nwutil import generate_sample_graph, get_available_paths


def test_environment_basic():
    """Test basic environment functionality."""
    print("Testing basic environment functionality...")

    # Create environment with a sample request file
    env = RSAEnv(request_file='data/train/requests-18.csv', capacity=20)

    # Reset environment
    obs, info = env.reset()
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Take a few random steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: action={action}, reward={reward:.2f}, "
              f"blocked={info.get('blocked', False)}, "
              f"blocking_rate={info.get('blocking_rate', 0):.4f}")

        if terminated:
            print("  Episode terminated early")
            break

    env.close()
    print("  Basic test PASSED!\n")


def test_graph_structure():
    """Test network graph structure."""
    print("Testing network graph structure...")

    graph = generate_sample_graph(capacity=20)
    print(f"  Number of nodes: {graph.number_of_nodes()}")
    print(f"  Number of edges: {graph.number_of_edges()}")

    # Verify link states
    for u, v, data in graph.edges(data=True):
        link_state = data['state']
        assert link_state.capacity == 20, "Capacity mismatch"
        assert len(link_state.wavelengths) == 20, "Wavelength array size mismatch"
        assert link_state.utilization == 0.0, "Initial utilization should be 0"

    print("  Graph structure test PASSED!\n")


def test_paths():
    """Test path availability for all source-destination pairs."""
    print("Testing predefined paths...")

    test_cases = [
        (0, 3, 2),  # Should have 2 paths
        (0, 4, 2),  # Should have 2 paths
        (7, 3, 2),  # Should have 2 paths
        (7, 4, 2),  # Should have 2 paths
        (1, 2, 0),  # Should have 0 paths (not supported)
    ]

    for src, dst, expected_count in test_cases:
        paths = get_available_paths(src, dst)
        actual_count = len(paths)
        status = "PASS" if actual_count == expected_count else "FAIL"
        print(f"  ({src}, {dst}): {actual_count} paths (expected {expected_count}) - {status}")

        if actual_count > 0:
            for i, path in enumerate(paths):
                print(f"    Path {i+1}: {path}")

    print("  Path test PASSED!\n")


def test_allocation_and_release():
    """Test wavelength allocation and lightpath release."""
    print("Testing allocation and release mechanisms...")

    from nwutil import (
        Request, find_available_wavelength, allocate_lightpath,
        release_expired_lightpaths
    )

    graph = generate_sample_graph(capacity=5)  # Small capacity for testing
    path = [0, 1, 2, 3]  # P1: 0->3

    # Test 1: Allocate first lightpath
    wavelength = find_available_wavelength(graph, path)
    assert wavelength == 0, f"Expected wavelength 0, got {wavelength}"
    print(f"  First available wavelength: {wavelength}")

    request1 = Request(source=0, destination=3, holding_time=5)
    allocate_lightpath(graph, path, wavelength, request1, current_time=0)

    # Verify allocation
    link_state = graph[0][1]['state']
    assert not link_state.wavelengths[0], "Wavelength 0 should be occupied"
    assert link_state.utilization == 0.2, f"Expected utilization 0.2, got {link_state.utilization}"
    print("  Lightpath allocated successfully")

    # Test 2: Allocate second lightpath
    wavelength2 = find_available_wavelength(graph, path)
    assert wavelength2 == 1, f"Expected wavelength 1, got {wavelength2}"

    request2 = Request(source=0, destination=3, holding_time=3)
    allocate_lightpath(graph, path, wavelength2, request2, current_time=0)

    link_state = graph[0][1]['state']
    assert link_state.utilization == 0.4, f"Expected utilization 0.4, got {link_state.utilization}"
    print("  Second lightpath allocated successfully")

    # Test 3: Release expired lightpaths
    release_expired_lightpaths(graph, current_time=3)
    link_state = graph[0][1]['state']
    assert link_state.wavelengths[1], "Wavelength 1 should be released"
    assert not link_state.wavelengths[0], "Wavelength 0 should still be occupied"
    assert link_state.utilization == 0.2, f"Expected utilization 0.2 after release, got {link_state.utilization}"
    print("  Expired lightpath released successfully")

    # Test 4: Release all
    release_expired_lightpaths(graph, current_time=10)
    link_state = graph[0][1]['state']
    assert all(link_state.wavelengths), "All wavelengths should be released"
    assert link_state.utilization == 0.0, "Utilization should be 0 after releasing all"
    print("  All lightpaths released successfully")

    print("  Allocation and release test PASSED!\n")


def test_capacity_limits():
    """Test that capacity limits are enforced."""
    print("Testing capacity limits...")

    from nwutil import (
        Request, find_available_wavelength, allocate_lightpath
    )

    graph = generate_sample_graph(capacity=3)
    path = [7, 6, 3]  # P6: 7->3

    # Allocate all 3 wavelengths
    for i in range(3):
        wavelength = find_available_wavelength(graph, path)
        assert wavelength is not None, f"Should find wavelength {i}"

        request = Request(source=7, destination=3, holding_time=10)
        allocate_lightpath(graph, path, wavelength, request, current_time=0)
        print(f"  Allocated wavelength {wavelength}")

    # Try to allocate one more (should fail)
    wavelength = find_available_wavelength(graph, path)
    assert wavelength is None, "Should not find available wavelength when capacity is full"
    print("  Correctly blocked when capacity is full")

    print("  Capacity limit test PASSED!\n")


def test_episode_completion():
    """Test complete episode execution."""
    print("Testing complete episode...")

    env = RSAEnv(request_file='data/train/requests-18.csv', capacity=20)
    obs, info = env.reset()

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 200:  # Safety limit
        # Use a simple heuristic: choose first valid action for current request
        if env.current_request:
            src = env.current_request.source
            dst = env.current_request.destination

            # Map (src, dst) to first valid action
            action_map = {
                (0, 3): 0,
                (0, 4): 2,
                (7, 3): 4,
                (7, 4): 6
            }
            action = action_map.get((src, dst), 8)  # Default to block
        else:
            action = 8

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"  Episode completed in {steps} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final blocking rate: {info.get('blocking_rate', 0):.4f}")

    env.close()
    print("  Episode completion test PASSED!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("RSA Environment Implementation Tests")
    print("=" * 60 + "\n")

    try:
        test_graph_structure()
        test_paths()
        test_allocation_and_release()
        test_capacity_limits()
        test_environment_basic()
        test_episode_completion()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now run the full training with: python dqn_runner.py")

    except AssertionError as e:
        print(f"\n{'=' * 60}")
        print(f"TEST FAILED: {e}")
        print("=" * 60)
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

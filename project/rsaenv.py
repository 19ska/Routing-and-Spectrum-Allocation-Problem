import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from nwutil import (
    Request, generate_sample_graph, get_available_paths,
    find_available_wavelength, allocate_lightpath,
    release_expired_lightpaths, get_network_state_vector
)


class RSAEnv(gym.Env):
    """
    Custom Gymnasium environment for Routing and Spectrum Allocation (RSA) problem.

    The environment simulates an optical network where requests arrive sequentially
    and need to be assigned to paths with available wavelengths.

    State Space:
        - Network link utilizations (12 links in the topology)
        - Current request information (source, destination, holding_time)

    Action Space:
        - Discrete actions representing path choices (0-7 for 8 possible paths)
        - Action 8 represents blocking the request

    Reward:
        - Positive reward for successfully allocating a request
        - Negative penalty for blocking a request
    """

    metadata = {'render_modes': []}

    def __init__(self, request_file=None, capacity=20):
        super(RSAEnv, self).__init__()

        self.capacity = capacity
        self.graph = generate_sample_graph(capacity=capacity)
        self.num_links = len(self.graph.edges())

        # Load requests from CSV file
        self.request_file = request_file
        self.requests = []
        if request_file:
            self._load_requests(request_file)

        # Simulation state
        self.current_time = 0
        self.request_idx = 0
        self.current_request = None
        self.blocked_count = 0
        self.total_requests = 0

        # Action space: 8 paths + 1 block action
        self.action_space = spaces.Discrete(9)

        # Observation space: link utilizations + request features
        # 12 link utilizations + 3 request features (src, dst, holding_time normalized)
        obs_dim = self.num_links + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def _load_requests(self, file_path):
        """Load requests from CSV file."""
        df = pd.read_csv(file_path)
        self.requests = []
        for _, row in df.iterrows():
            self.requests.append(
                Request(
                    source=int(row['source']),
                    destination=int(row['destination']),
                    holding_time=int(row['holding_time'])
                )
            )

    def _get_observation(self):
        """
        Generate the observation vector.
        Consists of:
        - Link utilizations (12 values)
        - Current request normalized features (3 values)
        """
        network_state = get_network_state_vector(self.graph)

        if self.current_request:
            # Normalize request features
            src_norm = self.current_request.source / 8.0  # Max node ID is 8
            dst_norm = self.current_request.destination / 8.0
            hold_norm = min(self.current_request.holding_time / 50.0, 1.0)  # Cap at 50
            request_features = [src_norm, dst_norm, hold_norm]
        else:
            request_features = [0.0, 0.0, 0.0]

        obs = np.array(network_state + request_features, dtype=np.float32)
        return obs

    def _get_path_for_action(self, action):
        """
        Map action index to path based on current request's src-dst pair.
        Returns None if action is invalid or represents blocking.
        """
        if action == 8:  # Block action
            return None

        if not self.current_request:
            return None

        src = self.current_request.source
        dst = self.current_request.destination
        paths = get_available_paths(src, dst)

        # Map action to path index
        # Actions 0-1: paths for (0,3)
        # Actions 2-3: paths for (0,4)
        # Actions 4-5: paths for (7,3)
        # Actions 6-7: paths for (7,4)

        path_map = {
            (0, 3): [0, 1],
            (0, 4): [2, 3],
            (7, 3): [4, 5],
            (7, 4): [6, 7]
        }

        valid_actions = path_map.get((src, dst), [])
        if action in valid_actions:
            path_idx = valid_actions.index(action)
            if path_idx < len(paths):
                return paths[path_idx]

        return None

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Integer representing the path choice or block action

        Returns:
            observation: Current state
            reward: Reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        reward = 0.0
        info = {'blocked': False, 'invalid_action': False}

        # Release expired lightpaths
        release_expired_lightpaths(self.graph, self.current_time)

        # Get the path for this action
        path = self._get_path_for_action(action)

        if path is None:
            # Block action or invalid action
            if action == 8:
                # Explicit block
                reward = -1.0
                self.blocked_count += 1
                info['blocked'] = True
            else:
                # Invalid action for current request
                reward = -2.0
                self.blocked_count += 1
                info['blocked'] = True
                info['invalid_action'] = True
        else:
            # Try to allocate on this path
            wavelength = find_available_wavelength(self.graph, path)

            if wavelength is not None:
                # Successfully allocate
                allocate_lightpath(self.graph, path, wavelength,
                                 self.current_request, self.current_time)
                reward = 1.0
                info['allocated'] = True
                info['path'] = path
                info['wavelength'] = wavelength
            else:
                # Path has no available wavelength - blocking
                reward = -1.0
                self.blocked_count += 1
                info['blocked'] = True

        # Move to next request
        self.current_time += 1
        self.request_idx += 1
        self.total_requests += 1

        # Check if episode is done
        terminated = self.request_idx >= len(self.requests)
        truncated = False

        if not terminated:
            self.current_request = self.requests[self.request_idx]
        else:
            self.current_request = None

        observation = self._get_observation()

        # Add blocking rate to info
        info['blocking_rate'] = self.blocked_count / self.total_requests if self.total_requests > 0 else 0.0

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (can include 'request_file')

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        # Handle options for request file
        if options and 'request_file' in options:
            self.request_file = options['request_file']
            self._load_requests(self.request_file)

        # Reset graph
        self.graph = generate_sample_graph(capacity=self.capacity)

        # Reset simulation state
        self.current_time = 0
        self.request_idx = 0
        self.blocked_count = 0
        self.total_requests = 0

        # Load first request
        if len(self.requests) > 0:
            self.current_request = self.requests[0]
        else:
            self.current_request = None

        observation = self._get_observation()
        info = {}

        return observation, info

    def render(self):
        """Render the environment (not implemented)."""
        pass

    def close(self):
        """Clean up resources."""
        pass

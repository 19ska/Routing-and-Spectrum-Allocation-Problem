# Implementation Summary

## Overview
This document provides a technical summary of the RSA (Routing and Spectrum Allocation) implementation using Deep Q-Networks.

## Files Created

### Core Implementation Files

1. **nwutil.py** (Extended)
   - Extended `LinkState` class with wavelength tracking
   - Added `wavelengths` array: Boolean list tracking availability of each wavelength
   - Added `lightpaths` dictionary: Maps wavelength index to (source, dest, expiry_time)
   - Implemented helper functions:
     - `get_available_paths()`: Returns pre-defined paths for source-destination pairs
     - `find_available_wavelength()`: First-fit wavelength allocation algorithm
     - `allocate_lightpath()`: Allocates wavelength on all links in a path
     - `release_expired_lightpaths()`: Releases lightpaths whose holding time expired
     - `get_network_state_vector()`: Converts graph to feature vector for DQN

2. **rsaenv.py** (New)
   - Custom Gymnasium environment implementing RSA problem
   - State space: 15-dimensional vector (12 link utilizations + 3 request features)
   - Action space: Discrete(9) - 8 paths + 1 block action
   - Reward function:
     - +1.0 for successful allocation
     - -1.0 for blocking
     - -2.0 for invalid action
   - Key methods:
     - `reset()`: Initialize episode with new request file
     - `step()`: Execute action, update state, return reward
     - `_get_observation()`: Generate state vector
     - `_get_path_for_action()`: Map action to path based on current request

3. **dqn_runner.py** (New)
   - Complete training and evaluation pipeline
   - `TrainingMetricsCallback`: Tracks episode rewards and blocking rates
   - `MultiFileEnvWrapper`: Cycles through training files for diverse experiences
   - `train_dqn_agent()`: Main training function with hyperparameter configuration
   - `evaluate_agent()`: Evaluates trained model on eval dataset
   - `plot_training_results()`: Generates learning curve and objective plots
   - `plot_evaluation_results()`: Generates evaluation performance plots
   - `main()`: Orchestrates training for both capacity=20 and capacity=10

### Supporting Files

4. **test_implementation.py** (New)
   - Comprehensive test suite for verifying implementation
   - Tests: graph structure, paths, allocation/release, capacity limits, episodes
   - Run before training to ensure correctness

5. **quick_eval.py** (New)
   - Command-line tool for quick model evaluation on single files
   - Usage: `python3 quick_eval.py <model_path> <request_file> [capacity]`

6. **requirements.txt** (New)
   - Lists all required Python packages
   - Install with: `pip install -r requirements.txt`

7. **README.md** (New)
   - Complete project documentation
   - Meets all submission requirements
   - Under 1500 words

8. **.gitignore** (New)
   - Excludes trained models, logs, and temporary files from Git

## Key Design Decisions

### 1. State Representation
**Decision**: Use link utilizations + normalized request features

**Rationale**:
- Compact representation (15D vs. hundreds of dims for full wavelength matrix)
- Captures essential information about network congestion
- Normalized features ensure all values in [0, 1] range
- Generalizes well across different request patterns

**Alternative Considered**: Include full wavelength availability matrix
- Rejected due to high dimensionality (12 links × capacity values)
- Would require more training time and larger networks

### 2. Action Space Design
**Decision**: Discrete actions (0-8) with invalid action penalties

**Rationale**:
- Simple discrete space works well with DQN
- Invalid actions penalized with -2.0 reward
- Agent learns to avoid invalid actions through experience

**Alternative Considered**: Action masking to prevent invalid actions
- Not natively supported by stable-baselines3 DQN
- Would require custom Q-network implementation
- Penalty approach is simpler and still effective

### 3. Reward Function
**Decision**: Sparse rewards (+1, -1, -2)

**Rationale**:
- Simple and interpretable
- Clear signal: allocate good, block bad
- Invalid actions more heavily penalized
- Directly aligned with objective (minimize blocking)

**Alternative Considered**: Dense rewards based on path length or utilization
- Could encourage specific path preferences
- Rejected to keep reward function simple and objective-aligned

### 4. First-Fit Wavelength Allocation
**Decision**: Always use lowest available wavelength index

**Rationale**:
- Required by problem specification
- Minimizes fragmentation in wavelength space
- Deterministic allocation simplifies debugging
- Standard practice in optical networks

### 5. Multi-File Training
**Decision**: Cycle through all 10,000 training files

**Rationale**:
- Ensures diverse training experiences
- Prevents overfitting to specific request sequences
- Simulates real-world traffic variability
- Better generalization to eval dataset

**Implementation**: `MultiFileEnvWrapper` rotates files on each `reset()`

### 6. Hyperparameter Selection

**Learning Rate (1e-4)**:
- Lower than default (1e-3) for stability
- Prevents oscillations in Q-value estimates
- Critical for convergence in capacity=10 scenario

**Buffer Size (50,000)**:
- Larger than default to store more diverse experiences
- Breaks correlation between consecutive samples
- Important given sequential nature of requests

**Exploration Fraction (0.3)**:
- Extended from default (0.1)
- Allows thorough exploration of path selection strategies
- Critical for discovering better policies

**Gamma (0.99)**:
- High discount factor appropriate for planning
- Considers long-term effects of allocation decisions
- Important because current allocations affect future availability

## Algorithmic Details

### Wavelength Continuity Constraint
- Ensured by `find_available_wavelength()` checking ALL links in path
- A wavelength is available only if free on every link
- No wavelength conversion allowed

### Capacity Constraint
- Enforced by `LinkState.wavelengths` array of size `capacity`
- Cannot allocate if all wavelengths occupied
- Utilization computed as: `occupied_wavelengths / capacity`

### Wavelength Conflict Constraint
- `lightpaths` dictionary ensures no overlapping allocations
- Wavelength marked unavailable when allocated
- Released only after holding time expires

### Time Management
- Logical time slots (not wall-clock time)
- Each request arrival = 1 time slot
- Expired lightpaths released at start of each slot
- Expiry time = current_time + holding_time

## Training Process

### Episode Structure
```
Episode = 100 requests from a single CSV file

For each request:
  1. Release expired lightpaths (check expiry_time <= current_time)
  2. Get observation (link utils + request features)
  3. Agent selects action (epsilon-greedy policy)
  4. Environment validates action
  5. If valid path: attempt allocation
  6. Compute reward
  7. Store transition in replay buffer
  8. Sample mini-batch and update Q-network
  9. current_time += 1
  10. Load next request
```

### Learning Mechanism
- **Experience Replay**: Store (s, a, r, s') in buffer, sample random mini-batches
- **Target Network**: Separate network for stable Q-value targets, updated every 1000 steps
- **Epsilon-Greedy**: Start with random exploration, gradually shift to learned policy
- **Temporal Difference Learning**: Update Q(s,a) based on observed reward + predicted future value

## Expected Results

### Capacity = 20
- **Training**: Blocking rate should decrease from ~0.5 to <0.05
- **Evaluation**: Blocking rate ~0.03-0.06, Objective ~0.94-0.97
- **Convergence**: Around episode 5,000-7,000

### Capacity = 10
- **Training**: Blocking rate should decrease from ~0.7 to 0.15-0.25
- **Evaluation**: Blocking rate ~0.20-0.30, Objective ~0.70-0.80
- **Convergence**: Around episode 7,000-9,000 (slower due to difficulty)

## Troubleshooting

### Common Issues

1. **Import Errors**: Install dependencies with `pip install -r requirements.txt`

2. **Memory Issues**: If training runs out of memory, reduce `buffer_size` in [dqn_runner.py](dqn_runner.py:100)

3. **Slow Training**: Expected to take 2-4 hours on CPU. Use GPU if available.

4. **Poor Performance**:
   - Check that first-fit allocation is working correctly
   - Verify expired lightpaths are being released
   - Ensure training files are loading correctly

5. **Action Space Issues**:
   - Verify path mapping in `_get_path_for_action()`
   - Check that invalid actions receive -2.0 penalty

## Testing Before Training

Always run tests first:
```bash
python3 test_implementation.py
```

This verifies:
- Graph structure is correct
- Path definitions match specification
- Allocation and release mechanisms work
- Capacity limits are enforced
- Episodes complete successfully

## Running Training

Full pipeline:
```bash
python3 dqn_runner.py
```

This will:
1. Train capacity=20 model (1M timesteps, ~10K episodes)
2. Evaluate on eval dataset
3. Generate 3 plots
4. Train capacity=10 model (1M timesteps, ~10K episodes)
5. Evaluate on eval dataset
6. Generate 3 plots
7. Save both models as .zip files

## Quick Evaluation

Test a trained model on a single file:
```bash
python3 quick_eval.py dqn_rsa_capacity20 data/eval/requests-0.csv 20
```

## File Locations After Training

- Models: `dqn_rsa_capacity20.zip`, `dqn_rsa_capacity10.zip`
- Training plots: `training_capacity{20|10}_{learning_curve|objective}.png`
- Eval plots: `eval_capacity{20|10}_objective.png`
- Logs: `tensorboard_logs/dqn_rsa_capacity{20|10}/`

## Code Quality

All files follow best practices:
- Clear, descriptive function and variable names
- Comprehensive docstrings
- Type hints where appropriate
- Modular design with single responsibility
- Extensive comments explaining key logic
- Error handling for edge cases

## Reproducibility

To reproduce results:
1. Clone repository
2. Install requirements: `pip install -r requirements.txt`
3. Run tests: `python3 test_implementation.py`
4. Run training: `python3 dqn_runner.py`
5. Models and plots will be generated automatically

Random seed is set in Gymnasium environment's `reset()` method for reproducibility.

## Extensions and Future Work

Possible improvements:
1. **Advanced Wavelength Assignment**: Implement other strategies (best-fit, most-used)
2. **Path Diversity**: Add more pre-defined paths for better load balancing
3. **Dynamic Routing**: Learn to generate paths instead of selecting from pre-defined set
4. **Multi-objective Optimization**: Balance blocking rate with path length
5. **Proactive Release**: Learn to release lightpaths early if blocking is imminent
6. **State Augmentation**: Add historical utilization or request arrival patterns
7. **Transfer Learning**: Use capacity=20 model to initialize capacity=10 training

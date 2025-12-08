# Quick Start Guide

## Installation (5 minutes)

```bash
cd /Users/skanda/Downloads/cs258-main/project

# Install required packages
pip install -r requirements.txt
```

## Verify Implementation (2 minutes)

```bash
# Run comprehensive tests
python3 test_implementation.py
```

Expected output: "ALL TESTS PASSED!"

## Training (2-4 hours)

```bash
# Train both models and generate all plots
python3 dqn_runner.py
```

This will:
- Train DQN with capacity=20 (1M timesteps)
- Train DQN with capacity=10 (1M timesteps)
- Evaluate both models on eval dataset
- Generate 6 plots (3 for each capacity)
- Save models as `dqn_rsa_capacity20.zip` and `dqn_rsa_capacity10.zip`

## Quick Test (30 seconds)

```bash
# After training, test a model on a single file
python3 quick_eval.py dqn_rsa_capacity20 data/eval/requests-0.csv 20
```

## Expected Output Files

After training completes, you will have:

### Trained Models
- `dqn_rsa_capacity20.zip` - Model for capacity=20
- `dqn_rsa_capacity10.zip` - Model for capacity=10

### Plots (6 total)
**Capacity=20:**
- `training_capacity20_learning_curve.png` - Episode rewards during training
- `training_capacity20_objective.png` - Objective (1 - blocking rate) during training
- `eval_capacity20_objective.png` - Objective on evaluation dataset

**Capacity=10:**
- `training_capacity10_learning_curve.png` - Episode rewards during training
- `training_capacity10_objective.png` - Objective (1 - blocking rate) during training
- `eval_capacity10_objective.png` - Objective on evaluation dataset

## Project Structure

```
project/
├── README.md                      # Main documentation (submit this)
├── nwutil.py                      # Network utilities (submit this)
├── rsaenv.py                      # Gymnasium environment (submit this)
├── dqn_runner.py                  # Training script (submit this)
├── dqn_rsa_capacity20.zip        # Trained model (submit this)
├── dqn_rsa_capacity10.zip        # Trained model (submit this)
├── requirements.txt               # Python dependencies
├── test_implementation.py         # Test suite
├── quick_eval.py                  # Quick evaluation tool
├── IMPLEMENTATION_SUMMARY.md      # Technical details
├── QUICKSTART.md                  # This file
└── data/
    ├── train/                     # 10,000 training files
    └── eval/                      # 1,000 evaluation files
```

## Submission Checklist

Before submitting, ensure you have:

- [ ] README.md (under 1500 words)
- [ ] rsaenv.py (custom environment)
- [ ] dqn_runner.py (training code)
- [ ] nwutil.py (utilities)
- [ ] dqn_rsa_capacity20.zip (trained model)
- [ ] dqn_rsa_capacity10.zip (trained model)
- [ ] All 6 plots embedded in README.md

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: data/train/..."
Make sure you're running from the project directory:
```bash
cd /Users/skanda/Downloads/cs258-main/project
```

### Training is too slow
- Expected: 2-4 hours on CPU for both models
- Reduce `total_timesteps` in [dqn_runner.py](dqn_runner.py) if needed (minimum 500K recommended)
- Use GPU if available (PyTorch with CUDA)

### Poor performance (high blocking rate)
- Verify tests pass: `python3 test_implementation.py`
- Check hyperparameters in [dqn_runner.py](dqn_runner.py:95-107)
- Increase `total_timesteps` for more training

## Next Steps

1. **Understand the Code**: Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
2. **Run Tests**: `python3 test_implementation.py`
3. **Start Training**: `python3 dqn_runner.py`
4. **Analyze Results**: Review generated plots
5. **Prepare Submission**: Push to GitHub repo

## GitHub Submission

```bash
# Initialize repo (if not already done)
git init
git add README.md rsaenv.py dqn_runner.py nwutil.py
git add dqn_rsa_capacity20.zip dqn_rsa_capacity10.zip
git add *.png  # Add all plot images
git commit -m "Complete RSA DQN implementation"
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Support

If you encounter issues:
1. Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details
2. Review error messages carefully
3. Verify all files are in correct locations
4. Ensure Python 3.8+ is installed
5. Check that data directory contains train/ and eval/ subdirectories

## Time Estimates

- **Setup**: 5 minutes
- **Testing**: 2 minutes
- **Training Capacity=20**: 1-2 hours
- **Training Capacity=10**: 1-2 hours
- **Total**: ~2-4 hours + setup

## Performance Benchmarks

Expected results on evaluation dataset:

| Capacity | Blocking Rate | Objective (1-BR) |
|----------|---------------|------------------|
| 20       | 0.03 - 0.06   | 0.94 - 0.97      |
| 10       | 0.20 - 0.30   | 0.70 - 0.80      |

If your results are within these ranges, your implementation is working correctly!

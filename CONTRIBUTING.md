# Contributing Guide

Quick guide for contributing to this project.

---

## Getting Started

### 1. Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/navsim-ijepa.git
cd navsim-ijepa/GTRS
```

### 2. Setup Environment
```bash
conda env create -f environment.yml
conda activate navsim
pip install -e ".[dev]"  # Includes pytest, black, mypy
```

### 3. Create Branch
```bash
git checkout -b feature/your-feature-name
```

---

## Code Style

### Format Before Committing
```bash
# Auto-format
black navsim/
isort navsim/

# Check
flake8 navsim/
mypy navsim/
```

### Example Code
```python
from typing import Dict
import torch
import torch.nn as nn


class MyAgent(AbstractAgent):
    """One-line description.
    
    Args:
        param1: Description.
        param2: Description.
    """
    
    def __init__(self, param1: int, param2: str = "default") -> None:
        super().__init__()
        self._param1 = param1  # Private attributes start with _
        self.public_attr = param2
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute predictions.
        
        Args:
            features: Input features.
            
        Returns:
            Dictionary with predictions.
        """
        # Implementation
        pass
```

**Rules**:
- Use type hints everywhere
- Google-style docstrings
- Max line length: 100 chars
- Double quotes for strings

---

## Testing

### Run Tests
```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=navsim

# Specific test
pytest tests/test_ijepa_agent.py -v
```

### Write Tests
```python
# tests/test_my_feature.py
def test_agent_forward():
    """Test forward pass works."""
    agent = IJEPAPlanningAgent()
    agent.initialize()
    
    features = {
        "image": torch.randn(1, 3, 512, 2048),
        "ego_history": torch.randn(1, 32)
    }
    
    output = agent.forward(features)
    
    assert "trajectory" in output
    assert output["trajectory"].shape == (1, 8, 3)
```

---

## Running Experiments

### 1. Create Config
```yaml
# configs/experiment/my_experiment.yaml
defaults:
  - /common/agent/ijepa_mlp_v2
  
experiment_name: my_experiment
trainer:
  max_epochs: 50
agent:
  learning_rate: 5e-5
```

### 2. Run Training
```bash
# Local (1 GPU)
python navsim/planning/script/run_training_dense.py \
    experiment=my_experiment \
    trainer.devices=1

# HPC (multiple GPUs)
sbatch scripts/train_multi_gpu.slurm
```

### 3. Document Results
Add to `summaries/EXPERIMENT_LOG.md`:
```markdown
## My Experiment (2025-11-06)

**Config**: Learning rate 5e-5, 50 epochs  
**Result**: PDMS = 0.8500  
**Notes**: 2% improvement over baseline
```

---

## Commit Messages

Format:
```
<type>: <short description>

<optional longer description>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code cleanup
- `test`: Add tests

Example:
```
feat: add transformer planning head

Replaced MLP with transformer decoder to capture long-range
dependencies in trajectory predictions.

- Added TransformerHead class
- Updated agent config
- Tests passing
```

---

## Pull Request

### Before Submitting
- [ ] Tests pass (`pytest tests/`)
- [ ] Code formatted (`black`, `isort`)
- [ ] No lint errors (`flake8`)
- [ ] Docstrings added
- [ ] Experiment documented (if applicable)

### Submit
1. Push to your fork
2. Create PR on GitHub
3. Fill in description
4. Wait for review

---

## Questions?

- **Issues**: Open a GitHub issue
- **Email**: [your.email@nyu.edu]

---

That's it! Keep it simple and ship experiments. 🚀

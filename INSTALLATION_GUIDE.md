# MSAI Environment Installation Guide

## Quick Start

### 1. Create Conda Environment

```bash
conda create -n msai python=3.11
conda activate msai
```

### 2. Install PyTorch with CUDA Support

**Important**: Install PyTorch first with the correct CUDA version:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

**For CPU-only** (if no GPU):
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
```

### 3. Install TorchAO for Advanced Quantization

```bash
pip install torchao==0.12.0
```

### 5. Install All Dependencies

```bash
cd /home/lence/msai
pip install -r requirements.txt
```

### 6. Install Individual Projects

For projects that need to be installed as packages (like UdaciSense):

```bash
# UdaciSense
cd /home/lence/msai/UdaciSense
pip install -e .

# Add other projects as needed
```

## Verification

Test your installation:

```python
import torch
import torchvision
import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
```

## Project-Specific Setup

### UdaciSense (Model Optimization)
```bash
cd /home/lence/msai/UdaciSense
pip install -e .
```

### GPT-2 Model Optimization
No additional setup required if main requirements are installed.

### CharityML
No additional setup required.

### Image Classification
No additional setup required.

### Customer Segmentation
No additional setup required.

## Troubleshooting

### Issue: "No module named 'utils'"
**Solution**: Make sure the project is installed in editable mode:
```bash
cd <project_directory>
pip install -e .
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size in your code or use CPU:
```python
device = torch.device('cpu')  # Force CPU usage
```

### Issue: Import errors after installation
**Solution**: Restart your Jupyter kernel or Python session.

### Issue: Conflicting package versions
**Solution**: Create a fresh conda environment:
```bash
conda deactivate
conda env remove -n msai
# Then follow installation steps again
```

## Package Management

### Adding New Packages

When adding a new package to any project:

1. Add it to `/home/lence/msai/requirements.txt`
2. Organize by category
3. Use `>=` for version constraints (unless specific version needed)
4. Update this guide if needed

Example:
```bash
# Add to appropriate category in requirements.txt
# Then update your environment:
pip install <package_name>>=<version>
```

### Updating Packages

To update all packages to latest compatible versions:
```bash
pip install -r requirements.txt --upgrade
```

To update specific package:
```bash
pip install --upgrade <package_name>
```

## Environment Export

To share your exact environment:
```bash
# Export conda environment
conda env export > environment.yml

# Export pip packages
pip freeze > requirements-frozen.txt
```

## Notes

- **Python Version**: 3.11 recommended for stability (especially with quantization)
- **PyTorch Version**: Pinned to 2.7.1 for compatibility with old quantization APIs
- **TorchAO Version**: 0.14.1 for modern quantization features
- **CUDA Version**: 12.8 for GPU support (optimized for RTX 3090)
- **Master Requirements**: All packages consolidated in `/home/lence/msai/requirements.txt`

## Support

For issues specific to:
- **UdaciSense**: Check `UdaciSense/PROJECT_GUIDE.md`
- **General ML**: See individual project READMEs

Last Updated: 2025-12-10

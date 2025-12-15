# MSAI Requirements Management

## 📁 File Structure

```
/home/lence/msai/
├── requirements.txt              # ⭐ MASTER file - All packages for all projects
├── INSTALLATION_GUIDE.md         # Complete installation instructions
└── <project>/
    └── requirements.txt          # Points to master file
```

## 🎯 Master Requirements File

**Location**: `/home/lence/msai/requirements.txt`

This is the **single source of truth** for all Python packages across all MSAI projects.

### Key Features:
- ✅ **Python 3.11** + **PyTorch 2.7.1** + **TorchAO 0.14.1** (optimized stack)
- ✅ Modern quantization support with TorchAO
- ✅ Supports all projects (UdaciSense, GPT-2, CharityML, etc.)
- ✅ Organized by category
- ✅ Version constraints for stability
- ✅ Platform-specific packages handled

## 🚀 Quick Setup

### For New Environment:

```bash
# 1. Create environment with Python 3.11
conda create -n msai python=3.11
conda activate msai

# 2. Install PyTorch 2.7.1 with CUDA support (do this first!)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 3. Install TorchAO for advanced quantization
pip install torchao==0.14.1

# 4. Install all dependencies
cd /home/lence/msai
pip install -r requirements.txt

# 5. Install specific project (if needed)
cd UdaciSense  # or any project
pip install -e .
```

### For Existing Environment:

```bash
cd /home/lence/msai
pip install -r requirements.txt
```

## ✅ Current Installation Status

**Verified Working** ✅:
- PyTorch 2.7.1+cu126
- Matplotlib 3.10.7
- NumPy 1.26.4
- Pandas 2.3.1
- Jupyter Lab 4.5.0
- All UdaciSense utils modules

## 📦 Adding New Packages

When you need a new package for **any** project:

### Step 1: Add to Master File
Edit `/home/lence/msai/requirements.txt`:

```python
# ==========================
# [Appropriate Category]
# ==========================
new-package>=X.Y.Z
```

### Step 2: Install
```bash
pip install new-package>=X.Y.Z
```

### Example:
```bash
# Add to requirements.txt under appropriate category
echo "opencv-python>=4.8.0  # Computer vision" >> requirements.txt

# Install
pip install opencv-python>=4.8.0
```

## 🔧 Package Categories in Master File

1. **PyTorch Ecosystem** - Deep learning framework
2. **Core Data Science** - NumPy, Pandas, SciPy, Scikit-learn
3. **Visualization** - Matplotlib, Seaborn, Plotly, Bokeh
4. **Jupyter & Interactive** - JupyterLab, IPython, kernels
5. **Deep Learning & Optimization** - TensorBoard, profiling tools
6. **Hugging Face** - Transformers, Datasets, Accelerate
7. **Model Compression** - Neural Compressor, Optimum
8. **Machine Learning** - XGBoost, AutoGluon
9. **Computer Vision** - Pillow, image processing
10. **NLP** - NLTK, text processing
11. **Data Handling** - Boto3, requests, data formats
12. **Utilities** - TQDM, psutil, dotenv

## 🎓 Version Strategy

- **Python**: 3.11 (recommended for stability)
- **PyTorch**: Pinned to `2.7.1` for stability with old quantization APIs
- **TorchAO**: Pinned to `0.14.1` (compatible with PyTorch 2.7.1)
- **Core packages**: `>=` with minimum version
- **Most packages**: Flexible versions (>=)
- **Breaking changes**: Pin specific version when needed

## 📊 Current Package Counts

The master requirements.txt includes:
- **~36 core packages** across all categories (including TorchAO)
- Supports **6+ different projects**
- Compatible with **Python 3.11** (recommended for stability)
- **CUDA 12.8** support for GPU acceleration (optimized for RTX 3090)
- **TorchAO 0.14.1** for modern quantization

## 🐛 Troubleshooting

### "No module named 'X'"
```bash
# Check if in requirements.txt
grep -i "package-name" /home/lence/msai/requirements.txt

# If not, add it and install
pip install package-name
```

### "No module named 'utils'" (UdaciSense)
```bash
cd /home/lence/msai/UdaciSense
pip install -e .
```

### Version conflicts
```bash
# Create fresh environment with Python 3.11
conda create -n msai-fresh python=3.11
conda activate msai-fresh
# Follow installation steps
```

### Check what's installed
```bash
pip list
pip show package-name  # Detailed info
```

## 📝 Maintenance

### Update All Packages
```bash
cd /home/lence/msai
pip install -r requirements.txt --upgrade
```

### Freeze Current Environment
```bash
pip freeze > requirements-frozen-$(date +%Y%m%d).txt
```

### Audit Dependencies
```bash
pip list --outdated
```

## 🔗 Related Files

- `/home/lence/msai/INSTALLATION_GUIDE.md` - Detailed setup guide
- `/home/lence/msai/UdaciSense/PROJECT_GUIDE.md` - UdaciSense specific guide
- Individual project READMEs

## 📅 Last Updated

**Date**: 2025-12-10
**Status**: ✅ Updated for Python 3.11 + PyTorch 2.7.1 + TorchAO 0.14.1
**Python**: 3.11 (recommended)
**PyTorch**: 2.7.1
**TorchAO**: 0.12.0
**CUDA**: 12.8

---

**Remember**: Always edit `/home/lence/msai/requirements.txt` when adding packages to any project. This keeps all projects in sync! 🎯

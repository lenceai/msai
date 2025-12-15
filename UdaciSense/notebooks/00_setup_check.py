"""
Setup verification script for UdaciSense notebooks.
Run this in a cell before your imports to verify everything is working.
"""

import sys
import os

# Add project root to path (so utils can be imported from anywhere)
# Find the project root by looking for UdaciSense directory
current_dir = os.getcwd()
project_root = None

# Check if we're in notebooks directory
if os.path.basename(current_dir) == 'notebooks':
    project_root = os.path.dirname(current_dir)
else:
    # Look for UdaciSense directory in current or parent
    if os.path.basename(current_dir) == 'UdaciSense':
        project_root = current_dir
    else:
        # Check parent directory
        parent_dir = os.path.dirname(current_dir)
        if os.path.basename(parent_dir) == 'UdaciSense':
            project_root = parent_dir

if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✅ Added {project_root} to Python path")
elif not project_root:
    print("⚠️  Could not find UdaciSense project root. Make sure you're running from UdaciSense/ or UdaciSense/notebooks/")

# Verify imports work
try:
    from utils import MAX_ALLOWED_ACCURACY_DROP, TARGET_INFERENCE_SPEEDUP, TARGET_MODEL_COMPRESSION
    print("✅ Utils constants imported successfully")
    print(f"   - Max Accuracy Drop: {MAX_ALLOWED_ACCURACY_DROP}")
    print(f"   - Target Speed Improvement: {TARGET_INFERENCE_SPEEDUP}")
    print(f"   - Target Size Reduction: {TARGET_MODEL_COMPRESSION}")
except ImportError as e:
    print(f"❌ Failed to import utils: {e}")
    sys.exit(1)

try:
    from utils.model import MobileNetV3_Household
    print("✅ Model utilities imported successfully")
except ImportError as e:
    print(f"❌ Failed to import model utilities: {e}")
    sys.exit(1)

try:
    from utils.data_loader import get_household_loaders
    print("✅ Data loader utilities imported successfully")
except ImportError as e:
    print(f"❌ Failed to import data loader: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} available")
    print(f"   - CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("❌ PyTorch not installed")
    sys.exit(1)

try:
    import matplotlib
    print(f"✅ Matplotlib {matplotlib.__version__} available")
except ImportError:
    print("❌ Matplotlib not installed")
    sys.exit(1)

try:
    import torchao
    print(f"✅ TorchAO {torchao.__version__} available")
except ImportError:
    print("⚠️  TorchAO not installed - some advanced quantization features may not work")

# Check CUDA version if available
if torch.cuda.is_available():
    print(f"   - CUDA version: {torch.version.cuda}")
    print(f"   - GPU: {torch.cuda.get_device_name(0)}")
    print(f"   - GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

print("\n🎉 All checks passed! You're ready to run the notebook.")

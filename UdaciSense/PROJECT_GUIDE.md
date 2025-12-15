# UdaciSense Model Optimization - Complete Project Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [Project Workflow](#project-workflow)
4. [Implementation Details](#implementation-details)
5. [Running Experiments](#running-experiments)
6. [Tips and Best Practices](#tips-and-best-practices)

## 📖 Project Overview

**Scenario**: You're a Machine Learning Engineer at UdaciHome working on the "UdaciSense" mobile app. Users complain about battery drain and slow performance. Your mission: optimize the vision model for mobile deployment.

**Requirements**:
- ✅ Reduce model size by **30%**
- ✅ Cut inference time by **40%**
- ✅ Maintain accuracy within **5%** of baseline

## 🚀 Setup Instructions

### 1. Environment Setup

```bash
cd /home/lence/msai/UdaciSense

# (Recommended) Install PyTorch first (GPU example: CUDA 12.8)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies (includes TorchAO 0.12.0)
pip install -r requirements.txt

# Install project as local package
pip install -e .
```

### 2. Verify Installation

```python
# In Python or Jupyter:
import torch
import torchvision
from utils.model import MobileNetV3_Household
from utils.data_loader import get_household_loaders

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 3. Directory Structure

The project will create these directories during execution:
```
UdaciSense/
├── data/                    # Dataset (auto-downloaded)
├── models/                  # Saved models
│   ├── baseline_mobilenet/ 
│   ├── post_training/
│   ├── in_training/
│   └── pipeline/
├── results/                 # Metrics and visualizations
│   ├── baseline_mobilenet/
│   ├── post_training/
│   ├── in_training/
│   └── pipeline/
└── ...
```

## 🔄 Project Workflow

### Phase 1: Baseline Analysis (01_baseline.ipynb)

**Objective**: Establish baseline metrics

**Tasks**:
1. Load and explore dataset (household objects from CIFAR-100)
2. Train MobileNetV3-Small from scratch or load pre-trained
3. Evaluate on test set:
   - Model size (MB)
   - Inference time (ms) 
   - Top-1 accuracy (%)
4. Analyze architecture for optimization opportunities

**Expected Results**:
- `models/baseline_mobilenet/checkpoints/model.pth`
- `results/baseline_mobilenet/metrics.json`
- Baseline performance visualizations

**Time Estimate**: 1-2 hours (depending on training)

### Phase 2: Compression Experiments (02_compression.ipynb)

**Objective**: Test individual compression techniques

**Available Techniques**:

#### Post-Training (Fast, No Retraining)
1. **Dynamic Quantization**
   - Quantizes Linear layers to INT8
   - Activations quantized during inference
   - ~4x size reduction, minimal accuracy loss

2. **Static Quantization**  
   - Quantizes weights and activations
   - Requires calibration data
   - Better accuracy than dynamic, more setup

3. **Pruning**
   - L1 Unstructured: Remove small magnitude weights
   - Global: Prune across all layers
   - Structured: Remove entire channels/filters
   - Test amounts: 0.3, 0.5, 0.7

4. **Graph Optimization**
   - TorchScript: JIT compilation
   - Torch FX: Operation fusion
   - Minimal size impact, speed improvements

#### In-Training (Slower, Better Results)
5. **Quantization-Aware Training (QAT)**
   - Simulates quantization during training
   - Model adapts to reduced precision
   - Best accuracy for quantized models
   - Config: Start epoch (e.g., 5), freeze BN epoch

6. **Gradual Pruning**
   - Progressive sparsity increase
   - Model adapts to weight removal
   - Schedules: linear, cubic, exponential
   - Config: Initial/final sparsity, epochs

7. **Knowledge Distillation**
   - Train smaller student from larger teacher
   - Config: Temperature (2-5), alpha (0.3-0.7)
   - Creates fundamentally smaller model

**Experiment Strategy**:
```python
# Example: Dynamic Quantization
quantization_type = "dynamic"
backend = "fbgemm"  # x86 CPU
device = torch.device('cpu')

quantized_model, results, name = apply_post_training_quantization(
    quantization_type, backend, device
)
```

**Expected Results**:
- 6-10 experiments (2 post-training + 2-3 in-training minimum)
- Comparison table showing trade-offs
- Insights on which techniques work best

**Time Estimate**: 
- Post-training: 30 min - 1 hour per technique
- In-training: 2-4 hours per technique (training time)

### Phase 3: Multi-Stage Pipeline (03_pipeline.ipynb)

**Objective**: Combine techniques for optimal results

**Pipeline Design Principles**:
1. **Sequencing Matters**:
   - Generally: Structural changes → Fine-tuning → Quantization
   - Example: Pruning → Fine-tune → Quantization
   
2. **Common Pipelines**:
   - **Conservative**: Graph Opt → Dynamic Quant
   - **Aggressive**: Prune (0.5) → Fine-tune → QAT
   - **Hybrid**: Distillation → Prune → Static Quant

3. **Pipeline Class Usage**:
```python
pipeline = OptimizationPipeline(
    name="pruning_quantization",
    baseline_model=baseline_model,
    train_loader=train_loader,
    test_loader=test_loader,
    class_names=class_names,
    input_size=input_size
)

# Add steps
pipeline.add_step("prune", apply_post_training_pruning, 
                  pruning_method="global_unstructured", amount=0.3)
pipeline.add_step("quantize", apply_dynamic_quantization)

# Run and visualize
optimized_model = pipeline.run(device=torch.device('cpu'))
pipeline.visualize_results(baseline_metrics)
```

**Iteration Strategy**:
1. Start simple (2 steps)
2. Check if requirements met
3. If not, add step or increase aggressiveness
4. If accuracy drops too much, reduce aggressiveness

**Expected Results**:
- 2-4 pipeline configurations tested
- At least one meeting all requirements
- Analysis of what works and why

**Time Estimate**: 2-4 hours

### Phase 4: Mobile Deployment (04_deployment.ipynb)

**Objective**: Package for mobile deployment

**Tasks**:
1. Convert best model to TorchScript
2. Apply mobile-specific optimizations
3. Verify output consistency
4. Measure mobile model size
5. Test inference on CPU (mobile simulation)
6. Document deployment considerations

**Mobile Conversion**:
```python
mobile_model = convert_model_for_mobile(
    optimized_model,
    input_size=input_size,
    mobile_optimize=True
)

# Save for deployment
torch.jit.save(mobile_model, "models/mobile/model.pt")

# Or lite interpreter (smaller)
mobile_model._save_for_lite_interpreter("models/mobile/model.ptl")
```

**Expected Results**:
- Mobile-ready model file (.pt or .ptl)
- Performance verification
- Deployment guide

**Time Estimate**: 30 minutes - 1 hour

### Phase 5: Final Report (report.md)

**Objective**: Document complete process and findings

**Report Sections**:
1. **Executive Summary**: Business-friendly overview
2. **Baseline Analysis**: Initial model characteristics
3. **Compression Techniques**: Individual technique results
4. **Pipeline Design**: Multi-stage approach and rationale
5. **Mobile Deployment**: Preparation and considerations
6. **Conclusions**: Achievements, insights, recommendations

**Time Estimate**: 1-2 hours

## 💡 Implementation Details

### Compression Technique Selection Guide

**When to Use Each Technique**:

| Technique | Best For | Pros | Cons |
|-----------|----------|------|------|
| Dynamic Quantization | Quick wins | Fast, no training | Limited compression |
| Static Quantization | Production deployment | Better accuracy than dynamic | Requires calibration |
| Post-Training Pruning | Exploration | Very fast | May need fine-tuning |
| QAT | Best quantized accuracy | Highest quality | Requires training |
| Gradual Pruning | High sparsity targets | Maintains accuracy | Long training |
| Distillation | Architectural change | Smaller model | Requires teacher |
| Graph Optimization | Speed improvements | Always beneficial | Minimal size impact |

### Hyperparameter Recommendations

**Quantization**:
- Backend: `fbgemm` for x86, `qnnpack` for ARM
- Calibration batches (static): 10-50

**Pruning**:
- Amount: Start at 0.3, increase to 0.5-0.7 if accuracy allows
- Method: Global usually best for overall sparsity
- Structured: Better hardware support but less flexible

**QAT**:
- Start epoch: 5-10 (train normally first)
- Freeze BN: Last 2-3 epochs
- Learning rate: Reduce when starting QAT

**Gradual Pruning**:
- Initial sparsity: 0.0
- Final sparsity: 0.5-0.7
- Start/end epoch: Leave ~5 epochs at start/end
- Schedule: Cubic (most common in literature)

**Knowledge Distillation**:
- Temperature: 2-5 (higher for more similar teacher/student)
- Alpha: 0.5 (balanced), 0.7 (emphasize teacher)
- Student size: 50-70% of teacher parameters

## 🎯 Tips and Best Practices

### General Tips

1. **Start Simple**: Test post-training techniques first
2. **Track Everything**: Save all experiment results
3. **Visualize Often**: Use provided plotting functions
4. **Validate Early**: Check accuracy after each step
5. **Use CPU for Mobile**: Mobile deployment is CPU-based

### Performance Tips

1. **Batch Size**: Use 128-256 for faster training
2. **Num Workers**: Set to CPU core count / 2
3. **Early Stopping**: Use patience=10 to save time
4. **Device Selection**: Use GPU for training, CPU for inference testing

### Debugging Tips

1. **Model Not Loading**: Check device compatibility
2. **Quantization Errors**: Ensure model is CPU and eval mode
3. **Pruning Not Removing**: Call prune.remove() to make permanent
4. **Low Accuracy**: Try less aggressive compression or fine-tuning

### Common Pitfalls

1. ❌ **Quantizing on GPU**: Quantization requires CPU
2. ❌ **Not Saving Intermediate Models**: Save after each pipeline step
3. ❌ **Ignoring Memory**: Monitor GPU/RAM usage during training
4. ❌ **Over-optimizing**: Don't sacrifice accuracy for marginal gains

## 📊 Expected Results Summary

**Typical Performance by Technique**:

| Technique | Size Reduction | Speed Improvement | Accuracy Loss |
|-----------|----------------|-------------------|---------------|
| Dynamic Quant | 60-75% | 30-40% | <1% |
| Static Quant | 60-75% | 40-50% | <2% |
| Pruning (50%) | 30-40% | 10-20% | 1-3% |
| QAT | 60-75% | 40-50% | <1% |
| Gradual Pruning (60%) | 40-50% | 15-25% | 2-4% |
| Distillation | 50-70% | 30-40% | 2-5% |
| Graph Opt | 0-5% | 10-20% | <0.5% |

**Common Successful Pipelines**:
1. **Pruning (30%) + Dynamic Quant**: 70% size, 45% speed, <2% accuracy
2. **QAT + Graph Opt**: 75% size, 50% speed, <1% accuracy
3. **Distillation + Static Quant**: 80% size, 55% speed, <3% accuracy

## 🆘 Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use CPU for that technique

### Issue: "Quantized model has poor accuracy"
**Solution**: Try QAT instead of post-training quantization

### Issue: "Pruning removes all weights"
**Solution**: Reduce pruning amount or use structured pruning

### Issue: "Pipeline doesn't meet requirements"
**Solution**: Increase technique aggressiveness or add more steps

### Issue: "Training takes too long"
**Solution**: Reduce epochs, use early stopping, or skip in-training techniques

## 📚 Additional Resources

- PyTorch Quantization Guide: https://pytorch.org/docs/stable/quantization.html
- Pruning Tutorial: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
- Knowledge Distillation: https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
- Mobile Deployment: https://pytorch.org/mobile/home/

## 🎓 Learning Outcomes

After completing this project, you will:
- ✅ Understand various model compression techniques
- ✅ Know when to apply each technique
- ✅ Be able to design multi-stage optimization pipelines
- ✅ Have experience with production model deployment
- ✅ Understand trade-offs between size, speed, and accuracy

Good luck with your optimization journey! 🚀

"""
UdaciSense Project: Post-Training Quantization Module

This module provides utilities for applying post-training quantization to PyTorch models,
supporting both static and dynamic quantization methods.
"""

import copy
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class QuantizableMobileNetV3_Household(nn.Module):
    """Lightweight wrapper to make an arbitrary model quantizable (eager/PTQ).

    We add Quant/DeQuant stubs at the model boundaries. This is the minimum
    required structure for eager static PTQ with `torch.ao.quantization.prepare/convert`.
    """

    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = original_model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self) -> None:
        """
        Fuse conv, bn, relu layers for better quantization results

        Args:
            model: Model to fuse
        """
        print("Fusing layers...")
        # For most torchvision models, module fusion requires model-specific patterns.
        # We keep this as a safe no-op by default, but if the wrapped model provides
        # a fuse_model() method we call it.
        if hasattr(self.model, "fuse_model"):
            try:
                self.model.fuse_model()
            except Exception as e:
                print(f"Warning: model.fuse_model() failed, continuing without fusion: {e}")
        

def quantize_model(
    model: nn.Module,
    calibration_data_loader: Optional[DataLoader] = None,
    calibration_num_batches: Optional[int] = None,
    quantization_type: str = "dynamic",
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply post-training quantization to a PyTorch model.
    
    Args:
        model: The original model to quantize
        calibration_data_loader: DataLoader for calibration data,
            required for static quantization
        calibration_num_batches: Number of batches to run calibration on
        quantization_type: Type of quantization to apply:
            - "dynamic": Dynamic quantization (weights are quantized, activations quantized during inference)
            - "static": Static quantization (weights and activations are pre-quantized)
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
            
    Returns:
        Quantized model
        
    Raises:
        ValueError: If an unsupported backend or quantization type is specified,
                   or if static quantization is requested without calibration data
    """
    # Verify backend
    if backend not in ["fbgemm", "qnnpack"]:
        raise ValueError("Backend must be either 'fbgemm' (x86) or 'qnnpack' (ARM)")
    
    # Create a copy of the model for quantization
    model_to_quantize = copy.deepcopy(model)
    
    # Set model to evaluation mode
    model_to_quantize.eval()
    
    # NOTE: Feel free to not implement all quantization types
    # Apply quantization based on type
    if quantization_type.lower() == "dynamic":
        return _apply_dynamic_quantization(model_to_quantize)
    elif quantization_type.lower() == "static":
        if calibration_data_loader is None:
            raise ValueError("Static quantization requires a calibration_data_loader")
        return _apply_static_quantization(model_to_quantize, calibration_data_loader, calibration_num_batches, backend)
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")

def _apply_dynamic_quantization(
    model: nn.Module
) -> nn.Module:
    """Apply dynamic quantization to a model.
    
    Dynamic quantization quantizes weights ahead of time but quantizes activations
    dynamically during inference.
    
    Args:
        model: Model to quantize (in eval mode)
        
    Returns:
        Dynamically quantized model
    """
    print("Applying dynamic quantization...")
    
    # Dynamic quantization - quantizes weights and dynamically quantizes activations
    # Apply to Linear and LSTM layers
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8  # Use 8-bit integers
    )
    
    print("Dynamic quantization complete")
    return quantized_model
                

def _apply_static_quantization(
    model: nn.Module,
    calibration_data_loader: DataLoader,
    calibration_num_batches: Optional[int] = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply static quantization to a model using provided calibration data.
    
    Static quantization quantizes both weights and activations ahead of time.
    
    Args:
        model: Model to quantize (in eval mode)
        calibration_data_loader: DataLoader for calibration data
        calibration_num_batches: Number of batches to use for calibration
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
        
    Returns:
        Statically quantized model
    """
    print("Applying static quantization...")
    
    # If calibration_num_batches is not specified, use all available batches
    if calibration_num_batches is None:
        calibration_num_batches = len(calibration_data_loader)
    
    # Set the backend
    torch.backends.quantized.engine = backend

    # Eager mode PTQ requires Quant/DeQuant stubs around the model.
    model_to_quantize = QuantizableMobileNetV3_Household(model).cpu().eval()

    # Configure quantization
    model_to_quantize.qconfig = torch.ao.quantization.get_default_qconfig(backend)

    # Optional fusion
    try:
        model_to_quantize.fuse_model()
    except Exception as e:
        print(f"Warning: Could not fuse model: {e}")

    # Prepare model for static quantization (insert observers)
    model_prepared = torch.ao.quantization.prepare(model_to_quantize, inplace=False)
    
    # Calibrate with representative dataset
    print(f"Calibrating with {calibration_num_batches} batches...")
    model_prepared.eval()
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(calibration_data_loader, desc="Calibration")):
            if batch_idx >= calibration_num_batches:
                break
            model_prepared(data.to(torch.device('cpu')))
    
    # Convert to quantized model
    quantized_model = torch.ao.quantization.convert(model_prepared, inplace=False)
    
    print("Static quantization complete")
    return quantized_model
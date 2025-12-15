"""
UdaciSense Project: Mobile Deployment Utilities

This module provides functions for converting models to mobile-friendly formats
and verifying their performance.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Union


def convert_model_for_mobile(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    mobile_optimize: bool = True
) -> torch.jit.ScriptModule:
    """
    Convert a PyTorch model to a mobile-friendly format using TorchScript.
    
    Args:
        model: PyTorch model to convert
        input_size: Shape of input tensor for tracing
        mobile_optimize: Whether to apply mobile-specific optimizations
        
    Returns:
        Mobile-optimized TorchScript model
    """
    print("Converting model for mobile deployment...")
    
    # Set model to evaluation mode
    model.eval()
    model = model.cpu()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(input_size)
    
    # Trace the model with TorchScript
    print("Tracing model with TorchScript...")
    try:
        traced_model = torch.jit.trace(model, dummy_input)
    except Exception as e:
        print(f"Error during tracing: {e}")
        raise
    
    # Optimize for inference
    print("Optimizing for inference...")
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    # Freeze the model
    print("Freezing model...")
    traced_model = torch.jit.freeze(traced_model)
    
    # Apply mobile-specific optimizations if requested
    if mobile_optimize:
        print("Applying mobile optimizations...")
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile
            traced_model = optimize_for_mobile(traced_model)
        except ImportError:
            print("Mobile optimizer not available, skipping mobile-specific optimizations")
    
    print("Model conversion complete!")
    return traced_model


def compare_model_outputs(
    model1: nn.Module,
    model2: nn.Module,
    input_tensor: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    device: torch.device = torch.device('cpu')
) -> bool:
    """
    Compare outputs of two models to verify consistency after conversion.
    
    Args:
        model1: First model
        model2: Second model
        input_tensor: Input tensor to test with
        rtol: Relative tolerance
        atol: Absolute tolerance
        device: Device to run on
        
    Returns:
        True if outputs are consistent, False otherwise
    """
    print("Comparing model outputs...")
    
    # Move models and input to specified device
    model1 = model1.to(device)
    model2 = model2.to(device)
    input_tensor = input_tensor.to(device)
    
    # Set models to evaluation mode
    model1.eval()
    model2.eval()
    
    # Get outputs from both models
    with torch.no_grad():
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)
    
    # Handle tuple outputs
    if isinstance(output1, tuple):
        output1 = output1[0]
    if isinstance(output2, tuple):
        output2 = output2[0]
    
    # Compare outputs
    is_close = torch.allclose(output1, output2, rtol=rtol, atol=atol)
    
    if is_close:
        print("✅ Model outputs are consistent!")
        max_diff = torch.max(torch.abs(output1 - output2))
        mean_diff = torch.mean(torch.abs(output1 - output2))
        print(f"   Max difference: {max_diff.item():.6f}")
        print(f"   Mean difference: {mean_diff.item():.6f}")
    else:
        max_diff = torch.max(torch.abs(output1 - output2))
        mean_diff = torch.mean(torch.abs(output1 - output2))
        print(f"❌ Model outputs differ!")
        print(f"   Max difference: {max_diff.item():.6f}")
        print(f"   Mean difference: {mean_diff.item():.6f}")
    
    return is_close


def get_model_file_size(model_path: str) -> Dict[str, float]:
    """
    Get the file size of a saved model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Dictionary with size information (size_bytes, size_mb)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Get file size in bytes
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    
    return {
        'size_bytes': size_bytes,
        'size_mb': size_mb
    }


def save_mobile_model(model: torch.jit.ScriptModule, save_path: str, use_lite_interpreter: bool = False) -> None:
    """
    Save a mobile-optimized model.
    
    Args:
        model: TorchScript model to save
        save_path: Path to save the model
        use_lite_interpreter: Whether to save with lite interpreter format (smaller, limited ops)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if use_lite_interpreter:
        print(f"Saving model with lite interpreter to {save_path}...")
        model._save_for_lite_interpreter(save_path)
    else:
        print(f"Saving model to {save_path}...")
        torch.jit.save(model, save_path)
    
    # Get and print file size
    size_info = get_model_file_size(save_path)
    print(f"Model saved successfully. Size: {size_info['size_mb']:.2f} MB")

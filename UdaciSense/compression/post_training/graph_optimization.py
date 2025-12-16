"""
Graph optimization utilities for PyTorch models.
Supports TorchScript and TorchFX optimizations with unified evaluation pipeline.
"""
import os
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.experimental.optimization import (
    fuse,
    remove_dropout, 
    optimize_for_inference
)
from typing import Dict, Any, Optional, Tuple, Literal, List, Union, Callable
import warnings
import json
import time

def optimize_model(
    model: nn.Module,
    optimization_method: Literal["torchscript", "torch_fx"] = "torchscript",
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = torch.device('cuda'),
    custom_options: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Optimize a model using PyTorch graph optimization techniques.
    
    Args:
        model: Model to optimize
        optimization_method: Optimization method to use
        input_shape: Input shape for tracing (e.g., (1, 3, 224, 224) for a single RGB image)
        device: Device to set the input and model on
        custom_options: Custom optimization options (optional)
        
    Returns:
        Optimized model
    """
    # Check if model is quantized (quantized models MUST run on CPU)
    is_quantized = False
    if not isinstance(model, torch.jit.ScriptModule):
        quantized_prefixes = (
            "torch.ao.nn.quantized",
            "torch.nn.quantized",
            "torch.ao.nn.intrinsic.quantized",
            "torch.nn.intrinsic.quantized",
        )
        try:
            for m in model.modules():
                mod = getattr(m, "__module__", "")
                if any(mod.startswith(p) for p in quantized_prefixes):
                    is_quantized = True
                    break
        except Exception:
            pass
    
    # Force CPU device for quantized models
    if is_quantized:
        print("Warning: Quantized model detected. Forcing CPU device (quantized ops only work on CPU).")
        device = torch.device('cpu')
    
    # Set model to evaluation mode
    model.eval()
    
    # Create a sample input tensor for tracing
    dummy_input = torch.randn(input_shape).to(device)
    
    # Move model to device (but quantized models should already be on CPU)
    if not is_quantized:
        model = model.to(device)
    else:
        # Ensure quantized model is on CPU
        model = model.cpu() if hasattr(model, 'cpu') else model
    
    # Apply optimization based on method
    if optimization_method == "torchscript":
        optimized_model = _optimize_with_torchscript(model, dummy_input, custom_options)    
    elif optimization_method == "torch_fx":
        optimized_model = _optimize_with_torch_fx(model, dummy_input, custom_options)
    else:
        raise ValueError(f"Unsupported optimization method: {optimization_method}")
        
    return optimized_model


def _optimize_with_torchscript(
    model: nn.Module,
    dummy_input: torch.Tensor,
    custom_options: Optional[Dict[str, Any]] = None
) -> torch.jit.ScriptModule:
    """
    Optimize model with TorchScript (JIT).
    
    Args:
        model: Model to optimize
        dummy_input: Sample input for tracing
        custom_options: Custom optimization options
        
    Returns:
        TorchScript optimized model
    """
    # Extract custom options with defaults
    if custom_options is None:
        custom_options = {}
    
    optimize_for_mobile = custom_options.get('optimize_for_mobile', False)
    # IMPORTANT:
    # - `torch.jit.optimize_for_inference()` can produce TorchScript modules that run but do NOT
    #   round-trip through `torch.jit.save/load` when the graph contains quantized packed params
    #   (observed in PyTorch 2.7: "required keyword attribute 'value' is undefined").
    # - For dynamically-quantized models, we skip `optimize_for_inference` to keep the model portable.
    skip_optimize_for_inference = custom_options.get('skip_optimize_for_inference', False)

    # Heuristic quantized detection (avoid importing utils into compression layer)
    def _looks_quantized(m: nn.Module) -> bool:
        try:
            for mod in m.modules():
                mod_name = getattr(mod, "__module__", "") or ""
                if mod_name.startswith(("torch.ao.nn.quantized", "torch.nn.quantized",
                                        "torch.ao.nn.intrinsic.quantized", "torch.nn.intrinsic.quantized")):
                    return True
        except Exception:
            pass
        return False
    
    print("Tracing model with TorchScript...")
    
    # Trace the model with JIT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced_model = torch.jit.trace(model, dummy_input)
    
    # NOTE: The recommended order is to freeze first (in eval mode), then run
    # optimize_for_inference(). In newer PyTorch versions, the module returned by
    # optimize_for_inference() may not expose a `.training` attribute, which
    # torch.jit.freeze() expects.
    print("Freezing model...")
    try:
        traced_model = traced_model.eval()
    except Exception:
        # Some ScriptModules may not fully implement nn.Module's train/eval API.
        # Freezing still requires eval-like behavior (no dropout/bn updates), and
        # our upstream code already sets the original model to eval().
        pass
    traced_model = torch.jit.freeze(traced_model)

    # Optimize for inference (optional).
    # Skip for quantized models to preserve save/load portability.
    if skip_optimize_for_inference or _looks_quantized(model):
        print("Skipping optimize_for_inference (quantized model or explicitly disabled).")
    else:
        print("Optimizing TorchScript model for inference...")
        traced_model = torch.jit.optimize_for_inference(traced_model)

    # Optional: mobile-specific optimizations.
    # We keep this behind a flag because it can change operator sets and is not
    # always desired for desktop/server inference.
    if optimize_for_mobile:
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile as _optimize_for_mobile
            traced_model = _optimize_for_mobile(traced_model)
        except Exception as e:
            print(f"Warning: optimize_for_mobile failed, continuing without it: {e}")
    
    print("TorchScript optimization complete")
    return traced_model


def _optimize_with_torch_fx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    custom_options: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Optimize model with Torch FX.
    
    Args:
        model: Model to optimize
        dummy_input: Sample input for tracing
        custom_options: Custom optimization options
        device: str
        
    Returns:
        FX-optimized model
    """
    # Extract custom options with defaults
    if custom_options is None:
        custom_options = {}
    
    print("Tracing model with Torch FX...")
    
    # Symbolically trace the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            traced_model = fx.symbolic_trace(model)
        except Exception as e:
            print(f"Warning: Could not symbolically trace model: {e}")
            print("Falling back to regular model...")
            return model
    
    # Apply optimizations
    print("Applying FX optimizations...")
    
    # Remove dropout layers (for inference)
    try:
        traced_model = remove_dropout(traced_model)
    except Exception as e:
        print(f"Could not remove dropout: {e}")
    
    # Fuse operations
    try:
        traced_model = fuse(traced_model)
    except Exception as e:
        print(f"Could not fuse operations: {e}")
    
    # Optimize for inference
    try:
        traced_model = optimize_for_inference(traced_model)
    except Exception as e:
        print(f"Could not optimize for inference: {e}")
    
    print("Torch FX optimization complete")
    return traced_model


def verify_model_equivalence(
    original_model: nn.Module,
    optimized_model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = torch.device('cpu'),
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> bool:
    """
    Verify equivalence between original and optimized models.
    
    Args:
        original_model: Original PyTorch model
        optimized_model: Optimized model
        input_shape: Shape of input tensor
        device: Device to run verification on
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        
    Returns:
        True if models are equivalent, False otherwise
    """
    # Set models to evaluation mode
    original_model.eval()
    try:
        optimized_model.eval()
    except Exception:
        # TorchScript modules produced by some passes (e.g., optimize_for_inference)
        # may not fully implement `nn.Module`'s training/eval toggles. They are
        # expected to already be in inference mode after export/optimization.
        pass
    
    # Create a random input tensor
    torch.manual_seed(0)  # For reproducibility
    input_tensor = torch.randn(input_shape, device=device)
    
    # Run inference with both models
    with torch.no_grad():
        original_output = original_model(input_tensor)
        optimized_output = optimized_model(input_tensor)
    
    # Compare outputs
    if isinstance(original_output, tuple):
        original_output = original_output[0]  # Use first output if model returns multiple
    
    if isinstance(optimized_output, tuple):
        optimized_output = optimized_output[0]
    
    # Check if the outputs are close
    is_close = torch.allclose(original_output, optimized_output, rtol=rtol, atol=atol)
    
    if is_close:
        print("Original and optimized models produce equivalent outputs.")
    else:
        max_diff = torch.max(torch.abs(original_output - optimized_output))
        mean_diff = torch.mean(torch.abs(original_output - optimized_output))
        print(f"Models differ. Max difference: {max_diff.item():.6f}, Mean difference: {mean_diff.item():.6f}")
    
    return is_close
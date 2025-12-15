"""
Post-Training Pruning techniques for model compression.
Supports both structured and unstructured pruning without retraining.
"""
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Any, Optional, Tuple, Literal, List, Union, Callable

from utils.compression import calculate_sparsity, find_prunable_modules, is_pruned
from utils.model import count_parameters, MobileNetV3_Household

def prune_model(
    model: nn.Module,
    pruning_method: Literal["l1_unstructured", "random_unstructured", "ln_structured", "global_unstructured"] = "l1_unstructured",
    amount: Union[float, int] = 0.3,
    modules_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
    custom_pruning_fn: Optional[Callable] = None,
    n: Optional[int] = None,
    dim: Optional[int] = None
) -> nn.Module:
    """
    Apply post-training pruning to a model.
    
    Args:
        model: Model to prune
        pruning_method: Type of pruning ("l1_unstructured", "random_unstructured", "ln_structured", "global_unstructured")
        amount: Amount to prune (fraction or absolute number)
        modules_to_prune: Specific modules to prune (optional, default is all Conv2d and Linear layers)
        custom_pruning_fn: Custom pruning function (optional)
        n: Order of the norm for ln_structured pruning (default=1)
        dim: Dimension along which to prune for structured pruning (default=0 for Conv2d output channels, 1 for Linear input features)
        
    Returns:
        Pruned model
    """
    
    # For the user-specified modules, or find all prunable modules (Conv2d and Linear)
    if modules_to_prune is None:
        modules_to_prune = find_prunable_modules(model)
    
    print(f"Pruning {len(modules_to_prune)} modules with method: {pruning_method}, amount: {amount}")
    
    # 1. Print initial sparsity
    initial_sparsity = calculate_sparsity(model)
    print(f"Initial model sparsity: {initial_sparsity:.2f}%")
    
    # NOTE: Feel free to implement one or all pruning methods
    if pruning_method == "l1_unstructured":
        _apply_unstructured_pruning(model, modules_to_prune, amount)
    
    elif pruning_method == "random_unstructured":
        _apply_random_unstructured_pruning(model, modules_to_prune, amount)
    
    elif pruning_method == "ln_structured":
        _apply_structured_pruning(model, modules_to_prune, amount, n, dim)
    
    elif pruning_method == "global_unstructured":
        _apply_global_pruning(model, modules_to_prune, amount)
    
    elif custom_pruning_fn is not None:
        # Apply custom pruning function
        custom_pruning_fn(model, modules_to_prune, amount)
        
    else:
        raise ValueError(f"Unsupported pruning method: {pruning_method}")
    
    # 3. Check that model is indeed pruned
    is_pruned(model)
    
    # 4. Print final sparsity
    final_sparsity = calculate_sparsity(model)
    print(f"Final model sparsity: {final_sparsity:.2f}%")
    
    # 5. Remove pruning reparameterization to make the pruning permanent
    print("Making pruning permanent...")
    for module, name in modules_to_prune:
        if hasattr(module, f'{name}_mask'):
            prune.remove(module, name)
    
    return model

def _apply_unstructured_pruning(
    model: nn.Module,
    modules_to_prune: List[Tuple[nn.Module, str]],
    amount: Union[float, int]
) -> nn.Module:
    """
    Apply unstructured (element-wise) L1 pruning to model.
    
    Args:
        model: Model to prune
        modules_to_prune: Modules to prune
        amount: Amount to prune
        
    Returns:
        Pruned model
    """
    print(f"Applying L1 unstructured pruning with amount={amount}")
    
    # Apply L1 unstructured pruning to each module
    for module, name in modules_to_prune:
        prune.l1_unstructured(module, name=name, amount=amount)
    
    return model

def _apply_random_unstructured_pruning(
    model: nn.Module,
    modules_to_prune: List[Tuple[nn.Module, str]],
    amount: Union[float, int]
) -> nn.Module:
    """
    Apply random unstructured pruning to model.
    
    Args:
        model: Model to prune
        modules_to_prune: Modules to prune
        amount: Amount to prune
        
    Returns:
        Pruned model
    """
    print(f"Applying random unstructured pruning with amount={amount}")
    
    # Apply random unstructured pruning to each module
    for module, name in modules_to_prune:
        prune.random_unstructured(module, name=name, amount=amount)
    
    return model

def _apply_structured_pruning(
    model: nn.Module,
    modules_to_prune: List[Tuple[nn.Module, str]],
    amount: Union[float, int],
    n: int = 1,
    dim: Optional[int] = None
) -> nn.Module:
    """
    Apply structured (channel/filter) pruning to model.
    
    Args:
        model: Model to prune
        modules_to_prune: Modules to prune
        amount: Amount to prune
        n: Order of the norm (default=1 for L1 norm)
        dim: Dimension along which to prune (if None, use default based on layer type)
        
    Returns:
        Pruned model
    """
    print(f"Applying Ln structured pruning with amount={amount}, n={n}")
    
    # Apply structured pruning to each module
    for module, name in modules_to_prune:
        # Determine dimension if not specified
        if dim is None:
            if isinstance(module, nn.Conv2d):
                prune_dim = 0  # Prune output channels for Conv2d
            elif isinstance(module, nn.Linear):
                prune_dim = 0  # Prune output features for Linear
            else:
                prune_dim = 0  # Default
        else:
            prune_dim = dim
        
        prune.ln_structured(module, name=name, amount=amount, n=n, dim=prune_dim)
    
    return model

def _apply_global_pruning(
    model: nn.Module,
    modules_to_prune: List[Tuple[nn.Module, str]],
    amount: Union[float, int]
) -> nn.Module:
    """
    Apply global pruning to model.
    
    Args:
        model: Model to prune
        modules_to_prune: Modules to prune
        amount: Amount to prune
        
    Returns:
        Pruned model
    """
    print(f"Applying global unstructured pruning with amount={amount}")
    
    # Apply global pruning across all specified modules
    prune.global_unstructured(
        modules_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    return model
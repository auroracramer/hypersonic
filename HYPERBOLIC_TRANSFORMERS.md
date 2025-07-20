# Hyperbolic Transformers Implementation

This document describes the implementation of transformers in hyperbolic space for the video prediction framework.

## Overview

The hyperbolic transformer implementation provides attention mechanisms that operate in the Poincaré ball model of hyperbolic space. This allows the model to capture hierarchical relationships in temporal data more effectively than standard Euclidean transformers.

## Components

### 1. HyperbolicMultiHeadAttention

Multi-head attention mechanism that computes attention scores using hyperbolic distances instead of dot products.

**Key Features:**
- Uses hyperbolic distance for computing attention scores
- Performs weighted averaging in hyperbolic space using Einstein midpoint
- Maintains all operations within the Poincaré ball

### 2. HyperbolicTransformerBlock

Complete transformer block with:
- Hyperbolic self-attention
- Hyperbolic feed-forward network
- Hyperbolic layer normalization
- Residual connections using Möbius addition

### 3. HyperbolicTransformer

Full transformer architecture with:
- Hyperbolic positional encoding
- Multiple transformer blocks
- Proper handling of hyperbolic space operations

### 4. ConvHyperbolicTransformer

Convolutional version for spatiotemporal modeling:
- Processes each spatial location independently through time
- Integrates with existing ConvGRU-based architecture
- Maintains spatial structure while modeling temporal dependencies

## Usage

### Basic Setup

To use hyperbolic transformers instead of ConvGRU, add the following arguments when running the main script:

```bash
python main.py \
  --hyperbolic \
  --use_transformer \
  --num_heads 8 \
  --transformer_layers 4 \
  --transformer_dropout 0.1 \
  --fp64_hyper \
  [other arguments...]
```

### Arguments

- `--use_transformer`: Enable transformer instead of ConvGRU
- `--num_heads`: Number of attention heads (default: 8)
- `--transformer_layers`: Number of transformer layers (default: 4)
- `--transformer_dropout`: Dropout rate in transformer (default: 0.1)
- `--hyperbolic`: Required for hyperbolic transformers
- `--fp64_hyper`: Use 64-bit precision for hyperbolic operations

### Example Training Script

See `scripts/train/train_kinetics_hyperbolic_transformer.sh` for a complete training example.

## Mathematical Foundation

### Hyperbolic Distance

Attention scores are computed using hyperbolic distance in the Poincaré ball:

```
d_hyp(x, y) = arccosh(1 + 2 * ||x - y||² / ((1 - ||x||²)(1 - ||y||²)))
```

### Einstein Midpoint

Weighted averaging in hyperbolic space uses the Einstein midpoint:

```
⊕_E w_i ⊙ x_i
```

where ⊕_E is the Einstein addition and ⊙ is hyperbolic scalar multiplication.

### Möbius Operations

All linear transformations and residual connections use Möbius operations:
- Möbius addition for residuals
- Möbius matrix-vector multiplication for linear layers
- Exponential and logarithmic maps for space transitions

## Implementation Details

### Numerical Stability

- Small initialization weights to avoid boundary issues
- Gradient clipping in hyperbolic operations
- Optional 64-bit precision for hyperbolic computations
- Careful handling of edge cases near the boundary

### Memory Efficiency

- Efficient computation of pairwise distances
- Optimized Einstein midpoint calculation
- Minimal memory overhead compared to standard transformers

### Integration

The hyperbolic transformer seamlessly integrates with the existing codebase:
- Replaces ConvGRU when `--use_transformer` is specified
- Compatible with all existing loss functions and training procedures
- Maintains the same input/output interface

## Performance Considerations

### When to Use Hyperbolic Transformers

- Temporal data with hierarchical structure
- Long sequences where traditional RNNs struggle
- When modeling complex dependencies between time steps
- Scenarios where the Poincaré ball's geometry is beneficial

### Computational Overhead

- Slightly higher computational cost than standard transformers
- Additional memory for hyperbolic operations
- Benefits from GPU acceleration for parallel attention computation

## Testing

Run the hyperbolic transformer tests:

```bash
python -m pytest tests/test_hyperbolic_transformer.py -v
python -m pytest tests/test_models.py::TestModel::test_model_forward_hyperbolic_transformer -v
```

## References

1. Ganea, O., Bécigneul, G., & Hofmann, T. (2018). Hyperbolic neural networks. NIPS.
2. Chami, I., Ying, Z., Ré, C., & Leskovec, J. (2019). Hyperbolic graph convolutional neural networks. NIPS.
3. Vaswani, A., et al. (2017). Attention is all you need. NIPS.

## Future Extensions

- Cross-attention between different spatial locations
- Hierarchical attention across multiple scales
- Integration with other hyperbolic neural network components
- Optimization for large-scale video datasets
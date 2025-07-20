"""
Tests for hyperbolic transformer implementations.
"""

import pytest
import torch
import numpy as np
from backbone.hyperbolic_transformer import (
    HyperbolicMultiHeadAttention,
    HyperbolicTransformerBlock,
    HyperbolicTransformer,
    ConvHyperbolicTransformer,
    HyperbolicLayerNorm,
    HyperbolicPositionalEncoding,
    HyperbolicActivation
)


class TestHyperbolicTransformer:
    """Test hyperbolic transformer components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.batch_size = 2
        self.seq_len = 8
        self.d_model = 64
        self.num_heads = 8
        self.d_ff = 256
        self.num_layers = 2
        self.k = -1.0
        self.fp64_hyper = False
    
    def test_hyperbolic_multi_head_attention_init(self):
        """Test HyperbolicMultiHeadAttention initialization."""
        attention = HyperbolicMultiHeadAttention(
            self.d_model, self.num_heads, k=self.k, fp64_hyper=self.fp64_hyper)
        
        assert attention.d_model == self.d_model
        assert attention.num_heads == self.num_heads
        assert attention.d_k == self.d_model // self.num_heads
        assert attention.k == self.k
    
    def test_hyperbolic_multi_head_attention_forward(self):
        """Test HyperbolicMultiHeadAttention forward pass."""
        attention = HyperbolicMultiHeadAttention(
            self.d_model, self.num_heads, k=self.k, fp64_hyper=self.fp64_hyper)
        
        # Create input in hyperbolic space
        x = torch.randn(self.batch_size, self.seq_len, self.d_model) * 0.1
        
        # Forward pass
        output, attn_weights = attention(x, x, x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
        assert attn_weights.shape == (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        
        # Check attention weights sum to 1
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-6)
    
    def test_hyperbolic_transformer_block_init(self):
        """Test HyperbolicTransformerBlock initialization."""
        block = HyperbolicTransformerBlock(
            self.d_model, self.num_heads, self.d_ff, k=self.k, fp64_hyper=self.fp64_hyper)
        
        assert block.k == self.k
        assert isinstance(block.self_attention, HyperbolicMultiHeadAttention)
        assert isinstance(block.norm1, HyperbolicLayerNorm)
        assert isinstance(block.norm2, HyperbolicLayerNorm)
    
    def test_hyperbolic_transformer_block_forward(self):
        """Test HyperbolicTransformerBlock forward pass."""
        block = HyperbolicTransformerBlock(
            self.d_model, self.num_heads, self.d_ff, k=self.k, fp64_hyper=self.fp64_hyper)
        
        # Create input in hyperbolic space
        x = torch.randn(self.batch_size, self.seq_len, self.d_model) * 0.1
        
        # Forward pass
        output, attn_weights = block(x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
        assert attn_weights.shape == (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
    
    def test_hyperbolic_transformer_init(self):
        """Test HyperbolicTransformer initialization."""
        transformer = HyperbolicTransformer(
            self.d_model, self.num_heads, self.num_layers, self.d_ff, 
            k=self.k, fp64_hyper=self.fp64_hyper)
        
        assert transformer.d_model == self.d_model
        assert len(transformer.layers) == self.num_layers
        assert isinstance(transformer.pos_encoding, HyperbolicPositionalEncoding)
    
    def test_hyperbolic_transformer_forward(self):
        """Test HyperbolicTransformer forward pass."""
        transformer = HyperbolicTransformer(
            self.d_model, self.num_heads, self.num_layers, self.d_ff, 
            k=self.k, fp64_hyper=self.fp64_hyper)
        
        # Create input in hyperbolic space
        x = torch.randn(self.batch_size, self.seq_len, self.d_model) * 0.1
        
        # Forward pass
        output, attention_weights = transformer(x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
        assert len(attention_weights) == self.num_layers
        for attn_weights in attention_weights:
            assert attn_weights.shape == (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
    
    def test_conv_hyperbolic_transformer_init(self):
        """Test ConvHyperbolicTransformer initialization."""
        conv_transformer = ConvHyperbolicTransformer(
            input_size=self.d_model, hidden_size=self.d_model, 
            num_heads=self.num_heads, num_layers=self.num_layers,
            k=self.k, fp64_hyper=self.fp64_hyper)
        
        assert conv_transformer.input_size == self.d_model
        assert conv_transformer.hidden_size == self.d_model
        assert isinstance(conv_transformer.transformer, HyperbolicTransformer)
    
    def test_conv_hyperbolic_transformer_forward(self):
        """Test ConvHyperbolicTransformer forward pass."""
        H, W = 4, 4
        conv_transformer = ConvHyperbolicTransformer(
            input_size=self.d_model, hidden_size=self.d_model, 
            num_heads=self.num_heads, num_layers=self.num_layers,
            k=self.k, fp64_hyper=self.fp64_hyper)
        
        # Create input [B, T, H, W, C]
        x = torch.randn(self.batch_size, self.seq_len, H, W, self.d_model) * 0.1
        
        # Forward pass
        output, attention_weights = conv_transformer(x)
        
        assert output.shape == (self.batch_size, self.seq_len, H, W, self.d_model)
        assert len(attention_weights) == H * W  # One set per spatial location
    
    def test_hyperbolic_layer_norm_init(self):
        """Test HyperbolicLayerNorm initialization."""
        layer_norm = HyperbolicLayerNorm(self.d_model, k=self.k, fp64_hyper=self.fp64_hyper)
        
        assert layer_norm.d_model == self.d_model
        assert layer_norm.k == self.k
        assert layer_norm.gamma.shape == (self.d_model,)
        assert layer_norm.beta.shape == (self.d_model,)
    
    def test_hyperbolic_layer_norm_forward(self):
        """Test HyperbolicLayerNorm forward pass."""
        layer_norm = HyperbolicLayerNorm(self.d_model, k=self.k, fp64_hyper=self.fp64_hyper)
        
        # Create input in hyperbolic space
        x = torch.randn(self.batch_size, self.seq_len, self.d_model) * 0.1
        
        # Forward pass
        output = layer_norm(x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
    
    def test_hyperbolic_positional_encoding_init(self):
        """Test HyperbolicPositionalEncoding initialization."""
        pos_encoding = HyperbolicPositionalEncoding(
            self.d_model, max_seq_len=1024, k=self.k, fp64_hyper=self.fp64_hyper)
        
        assert pos_encoding.d_model == self.d_model
        assert pos_encoding.k == self.k
        assert pos_encoding.pe.shape[1] == self.d_model
    
    def test_hyperbolic_positional_encoding_forward(self):
        """Test HyperbolicPositionalEncoding forward pass."""
        pos_encoding = HyperbolicPositionalEncoding(
            self.d_model, max_seq_len=1024, k=self.k, fp64_hyper=self.fp64_hyper)
        
        # Create input in hyperbolic space
        x = torch.randn(self.batch_size, self.seq_len, self.d_model) * 0.1
        
        # Forward pass
        output = pos_encoding(x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
    
    def test_hyperbolic_activation_init(self):
        """Test HyperbolicActivation initialization."""
        activation = HyperbolicActivation(k=self.k)
        
        assert activation.k == self.k
    
    def test_hyperbolic_activation_forward(self):
        """Test HyperbolicActivation forward pass."""
        activation = HyperbolicActivation(k=self.k)
        
        # Create input in hyperbolic space
        x = torch.randn(self.batch_size, self.seq_len, self.d_model) * 0.1
        
        # Forward pass
        output = activation(x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)


if __name__ == "__main__":
    pytest.main([__file__])
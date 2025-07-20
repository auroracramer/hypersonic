"""
Hyperbolic Transformer implementations for video prediction in hyperbolic space.
This module provides hyperbolic versions of transformer components including
self-attention, multi-head attention, and full transformer blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
import geoopt.manifolds.stereographic.math as gmath
from torch.cuda.amp import autocast
from .hyrnn_nets import MobiusLinear, mobius_linear


class HyperbolicMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism in hyperbolic space using the Poincaré ball model.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1, k=-1.0, fp64_hyper=True):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.k = torch.tensor(k)
        self.fp64_hyper = fp64_hyper
        
        # Hyperbolic linear projections for Q, K, V
        self.w_q = MobiusLinear(d_model, d_model, hyperbolic_input=True, 
                               hyperbolic_bias=True, k=k, fp64_hyper=fp64_hyper)
        self.w_k = MobiusLinear(d_model, d_model, hyperbolic_input=True, 
                               hyperbolic_bias=True, k=k, fp64_hyper=fp64_hyper)
        self.w_v = MobiusLinear(d_model, d_model, hyperbolic_input=True, 
                               hyperbolic_bias=True, k=k, fp64_hyper=fp64_hyper)
        self.w_o = MobiusLinear(d_model, d_model, hyperbolic_input=True, 
                               hyperbolic_bias=True, k=k, fp64_hyper=fp64_hyper)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def hyperbolic_attention_scores(self, q, k):
        """
        Compute attention scores in hyperbolic space using hyperbolic distance.
        """
        # q: [batch_size, num_heads, seq_len, d_k]
        # k: [batch_size, num_heads, seq_len, d_k]
        
        batch_size, num_heads, seq_len, d_k = q.shape
        
        # Compute pairwise hyperbolic distances
        # Reshape for distance computation
        q_expanded = q.unsqueeze(3)  # [batch, heads, seq_len, 1, d_k]
        k_expanded = k.unsqueeze(2)  # [batch, heads, 1, seq_len, d_k]
        
        # Compute hyperbolic distances between all pairs
        distances = gmath.dist(q_expanded, k_expanded, k=self.k, dim=-1)
        
        # Convert distances to attention scores (negative distance for similarity)
        attention_scores = -distances * self.scale
        
        return attention_scores
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.shape
        
        # Linear projections in hyperbolic space
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores using hyperbolic distances
        attention_scores = self.hyperbolic_attention_scores(Q, K)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        # For hyperbolic space, we need to use gyromidpoint for weighted averaging
        attended_values = self.hyperbolic_weighted_sum(V, attention_weights)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.w_o(attended_values)
        
        return output, attention_weights
    
    def hyperbolic_weighted_sum(self, values, weights):
        """
        Compute weighted sum in hyperbolic space using gyromidpoint.
        """
        batch_size, num_heads, seq_len, d_k = values.shape
        
        # Initialize output
        output = torch.zeros_like(values)
        
        for i in range(seq_len):
            # For each position, compute weighted combination of all values
            weighted_vals = []
            weight_vals = []
            
            for j in range(seq_len):
                if weights[..., i, j].sum() > 1e-10:  # Avoid zero weights
                    weighted_vals.append(values[..., j, :])
                    weight_vals.append(weights[..., i, j].unsqueeze(-1))
            
            if weighted_vals:
                # Stack values and weights
                stacked_vals = torch.stack(weighted_vals, dim=-2)  # [..., seq_len, d_k]
                stacked_weights = torch.stack(weight_vals, dim=-2)  # [..., seq_len, 1]
                
                # Normalize weights
                stacked_weights = stacked_weights / (stacked_weights.sum(dim=-2, keepdim=True) + 1e-10)
                
                # Compute hyperbolic weighted average using Einstein midpoint
                output[..., i, :] = self.einstein_midpoint(stacked_vals, stacked_weights)
            else:
                # If all weights are zero, use zero vector mapped to hyperbolic space
                output[..., i, :] = gmath.expmap0(torch.zeros_like(values[..., i, :]), k=self.k)
        
        return output
    
    def einstein_midpoint(self, points, weights):
        """
        Compute Einstein midpoint (weighted average) in hyperbolic space.
        """
        # Map to tangent space at origin
        tangent_points = gmath.logmap0(points, k=self.k)
        
        # Compute weighted average in tangent space
        weighted_avg = (tangent_points * weights).sum(dim=-2)
        
        # Map back to hyperbolic space
        result = gmath.expmap0(weighted_avg, k=self.k)
        
        return result


class HyperbolicTransformerBlock(nn.Module):
    """
    A complete transformer block in hyperbolic space.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, k=-1.0, fp64_hyper=True):
        super().__init__()
        
        self.k = torch.tensor(k)
        self.fp64_hyper = fp64_hyper
        
        # Multi-head attention
        self.self_attention = HyperbolicMultiHeadAttention(
            d_model, num_heads, dropout, k, fp64_hyper)
        
        # Feed-forward network in hyperbolic space
        self.feed_forward = nn.Sequential(
            MobiusLinear(d_model, d_ff, hyperbolic_input=True, 
                        hyperbolic_bias=True, k=k, fp64_hyper=fp64_hyper),
            HyperbolicActivation(k=k),
            nn.Dropout(dropout),
            MobiusLinear(d_ff, d_model, hyperbolic_input=True, 
                        hyperbolic_bias=True, k=k, fp64_hyper=fp64_hyper)
        )
        
        # Layer normalization in hyperbolic space
        self.norm1 = HyperbolicLayerNorm(d_model, k=k, fp64_hyper=fp64_hyper)
        self.norm2 = HyperbolicLayerNorm(d_model, k=k, fp64_hyper=fp64_hyper)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.hyperbolic_residual_connection(x, attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.hyperbolic_residual_connection(x, ff_output)
        x = self.norm2(x)
        
        return x, attn_weights
    
    def hyperbolic_residual_connection(self, x, sublayer_output):
        """
        Residual connection in hyperbolic space using Möbius addition.
        """
        return gmath.mobius_add(x, self.dropout(sublayer_output), k=self.k)


class HyperbolicActivation(nn.Module):
    """
    Hyperbolic activation function that operates in the tangent space.
    """
    
    def __init__(self, activation=nn.ReLU(), k=-1.0):
        super().__init__()
        self.activation = activation
        self.k = torch.tensor(k)
    
    def forward(self, x):
        # Map to tangent space
        tangent_x = gmath.logmap0(x, k=self.k)
        
        # Apply activation in tangent space
        activated = self.activation(tangent_x)
        
        # Map back to hyperbolic space
        return gmath.expmap0(activated, k=self.k)


class HyperbolicLayerNorm(nn.Module):
    """
    Layer normalization in hyperbolic space.
    """
    
    def __init__(self, d_model, eps=1e-6, k=-1.0, fp64_hyper=True):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.k = torch.tensor(k)
        self.fp64_hyper = fp64_hyper
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
        # Hyperbolic bias
        self.ball = geoopt.PoincareBall(c=self.k.abs())
        self.hyperbolic_bias = geoopt.ManifoldParameter(
            torch.zeros(d_model), manifold=self.ball)
        with torch.no_grad():
            self.hyperbolic_bias.set_(gmath.expmap0(
                torch.normal(0, 0.01, size=(d_model,)), k=self.k))
    
    def forward(self, x):
        if self.fp64_hyper:
            x = x.double()
        
        with autocast(enabled=False):
            # Map to tangent space for normalization
            tangent_x = gmath.logmap0(x, k=self.k)
            
            # Standard layer normalization in tangent space
            mean = tangent_x.mean(dim=-1, keepdim=True)
            std = tangent_x.std(dim=-1, keepdim=True)
            normalized = (tangent_x - mean) / (std + self.eps)
            
            # Scale and shift
            normalized = self.gamma * normalized + self.beta
            
            # Map back to hyperbolic space
            result = gmath.expmap0(normalized, k=self.k)
            
            # Add hyperbolic bias
            result = gmath.mobius_add(result, self.hyperbolic_bias.unsqueeze(0).expand_as(result), k=self.k)
            
            return result.float() if not self.fp64_hyper else result


class HyperbolicTransformer(nn.Module):
    """
    Complete hyperbolic transformer for temporal modeling.
    """
    
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_len=1024, 
                 dropout=0.1, k=-1.0, fp64_hyper=True):
        super().__init__()
        
        self.d_model = d_model
        self.k = torch.tensor(k)
        self.fp64_hyper = fp64_hyper
        
        # Positional encoding in hyperbolic space
        self.pos_encoding = HyperbolicPositionalEncoding(
            d_model, max_seq_len, k=k, fp64_hyper=fp64_hyper)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            HyperbolicTransformerBlock(d_model, num_heads, d_ff, dropout, k, fp64_hyper)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        attention_weights = []
        
        # Pass through transformer blocks
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights


class HyperbolicPositionalEncoding(nn.Module):
    """
    Positional encoding in hyperbolic space.
    """
    
    def __init__(self, d_model, max_seq_len=1024, k=-1.0, fp64_hyper=True):
        super().__init__()
        self.d_model = d_model
        self.k = torch.tensor(k)
        self.fp64_hyper = fp64_hyper
        
        # Create positional encoding in Euclidean space first
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Scale down to avoid boundary issues
        pe = pe * 0.1
        
        # Map to hyperbolic space
        pe_hyperbolic = gmath.expmap0(pe, k=self.k)
        
        self.register_buffer('pe', pe_hyperbolic)
    
    def forward(self, x):
        seq_len = x.size(1)
        pos_encoding = self.pe[:seq_len].unsqueeze(0).expand_as(x)
        
        if self.fp64_hyper:
            pos_encoding = pos_encoding.double()
        
        # Add positional encoding using Möbius addition
        return gmath.mobius_add(x, pos_encoding, k=self.k)


class ConvHyperbolicTransformer(nn.Module):
    """
    Convolutional Hyperbolic Transformer for spatiotemporal modeling.
    This combines convolutional operations with hyperbolic transformers.
    """
    
    def __init__(self, input_size, hidden_size, num_heads=8, num_layers=6, 
                 dropout=0.1, k=-1.0, fp64_hyper=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = torch.tensor(k)
        self.fp64_hyper = fp64_hyper
        
        # Input projection to hyperbolic space
        self.input_projection = MobiusLinear(
            input_size, hidden_size, hyperbolic_input=False, 
            hyperbolic_bias=True, k=k, fp64_hyper=fp64_hyper)
        
        # Hyperbolic transformer
        self.transformer = HyperbolicTransformer(
            hidden_size, num_heads, num_layers, hidden_size * 4, 
            dropout=dropout, k=k, fp64_hyper=fp64_hyper)
        
        # Output projection
        self.output_projection = MobiusLinear(
            hidden_size, hidden_size, hyperbolic_input=True, 
            hyperbolic_bias=True, k=k, fp64_hyper=fp64_hyper)
    
    def forward(self, x, mask=None):
        # x: [B, T, H, W, C] -> [B, T, H*W, C]
        B, T, H, W, C = x.shape
        
        # Reshape for sequence modeling
        x = x.view(B, T, H * W, C)
        
        # Process each spatial location independently
        outputs = []
        all_attention_weights = []
        
        for spatial_idx in range(H * W):
            # Extract temporal sequence for this spatial location
            spatial_seq = x[:, :, spatial_idx, :]  # [B, T, C]
            
            # Project to hyperbolic space
            spatial_seq_hyp = self.input_projection(spatial_seq)
            
            # Apply transformer
            transformed_seq, attention_weights = self.transformer(spatial_seq_hyp, mask)
            
            # Apply output projection
            output_seq = self.output_projection(transformed_seq)
            
            outputs.append(output_seq)
            all_attention_weights.append(attention_weights)
        
        # Combine outputs
        output = torch.stack(outputs, dim=2)  # [B, T, H*W, hidden_size]
        output = output.view(B, T, H, W, self.hidden_size)
        
        return output, all_attention_weights
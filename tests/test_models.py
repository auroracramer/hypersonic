"""Tests for models module."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

import models


class TestModel:
    """Test cases for the Model class."""

    def test_model_initialization(self):
        """Test that Model can be initialized with basic arguments."""
        # Mock args object
        args = Mock()
        args.seq_len = 10
        args.img_dim = 224
        args.network_feature = 'resnet18'
        args.not_track_running_stats = False
        args.feature_dim = -1
        args.hyperbolic = False
        args.hyperbolic_version = 1
        args.num_seq = 5
        args.pred_step = 3
        args.distance = 'regular'
        args.cross_gpu_score = False
        args.use_labels = False
        args.hierarchical_labels = False
        args.pred_future = False
        args.early_action = False
        args.early_action_self = False
        args.dataset = 'kinetics'
        args.device = 'cpu'

        # Test model creation
        model = models.Model(args)
        
        assert isinstance(model, nn.Module)
        assert model.last_duration == 3  # ceil(10/4)
        assert model.last_size == 7  # ceil(224/32)

    def test_model_forward_basic(self):
        """Test basic forward pass."""
        # Mock args
        args = Mock()
        args.seq_len = 8
        args.img_dim = 224
        args.network_feature = 'resnet18'
        args.not_track_running_stats = False
        args.feature_dim = 512
        args.hyperbolic = False
        args.hyperbolic_version = 1
        args.num_seq = 4
        args.pred_step = 2
        args.distance = 'regular'
        args.cross_gpu_score = False
        args.use_labels = False
        args.hierarchical_labels = False
        args.pred_future = False
        args.early_action = False
        args.early_action_self = False
        args.dataset = 'kinetics'
        args.device = 'cpu'

        model = models.Model(args)
        
        # Create dummy input
        batch_size = 2
        channels = 3
        seq_len = 8
        height = 224
        width = 224
        
        # Input shape: [B, N, C, SL, H, W]
        dummy_input = torch.randn(batch_size, args.num_seq, channels, seq_len, height, width)
        
        # Test forward pass
        with torch.no_grad():
            output = model(dummy_input)
            
        assert isinstance(output, (tuple, list))
        assert len(output) >= 2  # Should return predictions and features
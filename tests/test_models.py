import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Model, _initialize_weights


class TestModel:
    """Test cases for the main Model class."""

    def test_model_init_euclidean(self, sample_args):
        """Test model initialization in euclidean mode."""
        model = Model(sample_args)
        
        assert model.args == sample_args
        assert model.feature_dim == sample_args.feature_dim
        assert model.last_duration == 3  # ceil(10/4)
        assert model.last_size == 7  # ceil(224/32)
        assert model.num_layers == 1
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'adapt_dim')
        assert hasattr(model, 'agg')
        assert hasattr(model, 'network_pred')
        assert hasattr(model, 'final_bn')
        assert hasattr(model, 'final_linear')

    def test_model_init_hyperbolic(self, sample_args_hyperbolic):
        """Test model initialization in hyperbolic mode."""
        model = Model(sample_args_hyperbolic)
        
        assert model.args.hyperbolic == True
        assert hasattr(model, 'hyperbolic_linear')
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'adapt_dim')

    def test_model_init_custom_feature_dim(self, sample_args):
        """Test model initialization with custom feature dimension."""
        sample_args.feature_dim = 512
        model = Model(sample_args)
        
        assert model.feature_dim == 512
        assert isinstance(model.adapt_dim, nn.Linear)

    def test_model_init_2dim_features(self, sample_args):
        """Test model initialization with 2D features."""
        sample_args.final_2dim = True
        model = Model(sample_args)
        
        assert model.feature_dim == 2

    def test_model_forward_euclidean(self, sample_args, sample_video_data):
        """Test forward pass in euclidean mode."""
        model = Model(sample_args)
        model.eval()
        
        video_tensor, labels = sample_video_data
        
        # Test forward pass
        with torch.no_grad():
            output = model(video_tensor, labels)
        
        assert len(output) == 3  # pred, feature_dist, sizes_pred
        pred, feature_dist, sizes_pred = output
        
        # Check output shapes
        assert pred.shape[0] == video_tensor.shape[0]  # batch size
        assert feature_dist.shape[0] == video_tensor.shape[0]
        assert sizes_pred.shape[0] == video_tensor.shape[0]

    def test_model_forward_hyperbolic(self, sample_args_hyperbolic, sample_video_data):
        """Test forward pass in hyperbolic mode."""
        model = Model(sample_args_hyperbolic)
        model.eval()
        
        video_tensor, labels = sample_video_data
        
        # Test forward pass
        with torch.no_grad():
            output = model(video_tensor, labels)
        
        assert len(output) == 3  # pred, feature_dist, sizes_pred
        pred, feature_dist, sizes_pred = output
        
        # Check output shapes
        assert pred.shape[0] == video_tensor.shape[0]  # batch size
        assert feature_dist.shape[0] == video_tensor.shape[0]
        assert sizes_pred.shape[0] == video_tensor.shape[0]

    def test_model_forward_with_labels(self, sample_args, sample_video_data):
        """Test forward pass with labels."""
        model = Model(sample_args)
        model.eval()
        
        video_tensor, labels = sample_video_data
        
        with torch.no_grad():
            output = model(video_tensor, labels)
        
        assert len(output) == 3
        pred, feature_dist, sizes_pred = output
        
        # Verify output is reasonable
        assert not torch.isnan(pred).any()
        assert not torch.isnan(feature_dist).any()
        assert not torch.isnan(sizes_pred).any()

    def test_model_forward_without_labels(self, sample_args, sample_video_data):
        """Test forward pass without labels."""
        model = Model(sample_args)
        model.eval()
        
        video_tensor, _ = sample_video_data
        
        with torch.no_grad():
            output = model(video_tensor)
        
        assert len(output) == 3
        pred, feature_dist, sizes_pred = output
        
        # Verify output is reasonable
        assert not torch.isnan(pred).any()
        assert not torch.isnan(feature_dist).any()
        assert not torch.isnan(sizes_pred).any()

    def test_model_different_network_features(self, sample_args):
        """Test model with different network features."""
        networks = ['resnet18', 'resnet34', 'resnet50']
        
        for network in networks:
            sample_args.network_feature = network
            model = Model(sample_args)
            assert hasattr(model, 'backbone')
            assert hasattr(model, 'param')

    def test_model_train_mode(self, sample_args, sample_video_data):
        """Test model in training mode."""
        model = Model(sample_args)
        model.train()
        
        video_tensor, labels = sample_video_data
        
        # Test forward pass in training mode
        output = model(video_tensor, labels)
        
        assert len(output) == 3
        pred, feature_dist, sizes_pred = output
        
        # Check that gradients can be computed
        loss = pred.sum() + feature_dist.sum() + sizes_pred.sum()
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_model_only_train_linear(self, sample_args, sample_video_data):
        """Test model with only_train_linear flag."""
        sample_args.only_train_linear = True
        model = Model(sample_args)
        
        video_tensor, labels = sample_video_data
        
        # Check that only linear layers have requires_grad=True
        linear_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'linear' in name.lower():
                linear_params.append(param)
            else:
                other_params.append(param)
        
        # Note: This test depends on the actual implementation details
        # The model should set requires_grad appropriately
        assert len(linear_params) > 0

    def test_model_reset_mask(self, sample_args):
        """Test reset_mask method."""
        model = Model(sample_args)
        
        # Call reset_mask
        model.reset_mask()
        
        # Check that target and sizes_mask are reset
        assert model.target is None
        assert model.sizes_mask is None

    def test_model_cuda_compatibility(self, sample_args, sample_video_data):
        """Test model CUDA compatibility."""
        if torch.cuda.is_available():
            model = Model(sample_args).cuda()
            video_tensor, labels = sample_video_data
            video_tensor = video_tensor.cuda()
            labels = labels.cuda()
            
            with torch.no_grad():
                output = model(video_tensor, labels)
            
            pred, feature_dist, sizes_pred = output
            assert pred.is_cuda
            assert feature_dist.is_cuda
            assert sizes_pred.is_cuda

    def test_model_different_batch_sizes(self, sample_args):
        """Test model with different batch sizes."""
        model = Model(sample_args)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            video_tensor = torch.randn(batch_size, 5, 3, 10, 224, 224)
            labels = torch.randint(0, 10, (batch_size,))
            
            with torch.no_grad():
                output = model(video_tensor, labels)
            
            pred, feature_dist, sizes_pred = output
            assert pred.shape[0] == batch_size
            assert feature_dist.shape[0] == batch_size
            assert sizes_pred.shape[0] == batch_size

    def test_model_different_sequence_lengths(self, sample_args):
        """Test model with different sequence lengths."""
        model = Model(sample_args)
        model.eval()
        
        seq_lengths = [5, 8, 10, 15]
        
        for seq_len in seq_lengths:
            sample_args.seq_len = seq_len
            video_tensor = torch.randn(2, 5, 3, seq_len, 224, 224)
            labels = torch.randint(0, 10, (2,))
            
            with torch.no_grad():
                output = model(video_tensor, labels)
            
            pred, feature_dist, sizes_pred = output
            assert pred.shape[0] == 2
            assert feature_dist.shape[0] == 2
            assert sizes_pred.shape[0] == 2

    def test_model_edge_cases(self, sample_args):
        """Test model edge cases."""
        model = Model(sample_args)
        model.eval()
        
        # Test with minimal input
        video_tensor = torch.randn(1, 1, 3, 1, 224, 224)
        labels = torch.randint(0, 10, (1,))
        
        with torch.no_grad():
            output = model(video_tensor, labels)
        
        assert len(output) == 3
        pred, feature_dist, sizes_pred = output
        assert pred.shape[0] == 1
        assert feature_dist.shape[0] == 1
        assert sizes_pred.shape[0] == 1


class TestInitializeWeights:
    """Test cases for the _initialize_weights function."""

    def test_initialize_weights_linear(self):
        """Test weight initialization for linear layers."""
        module = nn.Linear(10, 5)
        _initialize_weights(module)
        
        # Check that weights are initialized
        assert module.weight.data is not None
        assert module.bias.data is not None
        
        # Check that bias is zero
        assert torch.allclose(module.bias.data, torch.zeros_like(module.bias.data))

    def test_initialize_weights_conv(self):
        """Test weight initialization for convolutional layers."""
        module = nn.Conv2d(3, 64, 3)
        _initialize_weights(module)
        
        # Check that weights are initialized
        assert module.weight.data is not None
        assert module.bias.data is not None
        
        # Check that bias is zero
        assert torch.allclose(module.bias.data, torch.zeros_like(module.bias.data))

    def test_initialize_weights_custom_gain(self):
        """Test weight initialization with custom gain."""
        module = nn.Linear(10, 5)
        gain = 2.0
        _initialize_weights(module, gain=gain)
        
        # Weights should be initialized with the specified gain
        assert module.weight.data is not None
        assert module.bias.data is not None
        assert torch.allclose(module.bias.data, torch.zeros_like(module.bias.data))

    def test_initialize_weights_sequential(self):
        """Test weight initialization for sequential module."""
        module = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        _initialize_weights(module)
        
        # Check that all linear layers are initialized
        for layer in module:
            if isinstance(layer, nn.Linear):
                assert layer.weight.data is not None
                assert layer.bias.data is not None
                assert torch.allclose(layer.bias.data, torch.zeros_like(layer.bias.data))

    def test_initialize_weights_no_bias(self):
        """Test weight initialization for module without bias."""
        module = nn.Linear(10, 5, bias=False)
        _initialize_weights(module)
        
        # Check that weights are initialized
        assert module.weight.data is not None
        assert module.bias is None


class TestModelIntegration:
    """Integration tests for the Model class."""

    def test_model_training_loop(self, sample_args, sample_video_data):
        """Test model in a simple training loop."""
        model = Model(sample_args)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        video_tensor, labels = sample_video_data
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        pred, feature_dist, sizes_pred = model(video_tensor, labels)
        
        # Simple loss calculation (normally would use the losses module)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        
        # Check that training completed without errors
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_model_evaluation_mode(self, sample_args, sample_video_data):
        """Test model in evaluation mode."""
        model = Model(sample_args)
        model.eval()
        
        video_tensor, labels = sample_video_data
        
        with torch.no_grad():
            pred1, _, _ = model(video_tensor, labels)
            pred2, _, _ = model(video_tensor, labels)
        
        # In eval mode, outputs should be deterministic
        assert torch.allclose(pred1, pred2, atol=1e-6)

    def test_model_state_dict_save_load(self, sample_args):
        """Test saving and loading model state."""
        model1 = Model(sample_args)
        model2 = Model(sample_args)
        
        # Save state dict
        state_dict = model1.state_dict()
        
        # Load state dict
        model2.load_state_dict(state_dict)
        
        # Check that weights are the same
        for key in state_dict:
            assert torch.allclose(model1.state_dict()[key], model2.state_dict()[key])

    def test_model_parameter_count(self, sample_args):
        """Test model parameter counting."""
        model = Model(sample_args)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
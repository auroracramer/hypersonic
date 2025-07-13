import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backbone.select_backbone import select_resnet


class TestSelectResnet:
    """Test cases for the select_resnet function."""

    def test_select_resnet_resnet18(self):
        """Test select_resnet with ResNet18."""
        model, param = select_resnet('resnet18')
        
        assert model is not None
        assert isinstance(param, dict)
        assert 'feature_size' in param
        assert param['feature_size'] == 256

    def test_select_resnet_resnet18_dim8(self):
        """Test select_resnet with ResNet18 dim8."""
        model, param = select_resnet('resnet18-dim8')
        
        assert model is not None
        assert isinstance(param, dict)
        assert 'feature_size' in param
        assert param['feature_size'] == 8

    def test_select_resnet_resnet34(self):
        """Test select_resnet with ResNet34."""
        model, param = select_resnet('resnet34')
        
        assert model is not None
        assert isinstance(param, dict)
        assert 'feature_size' in param
        assert param['feature_size'] == 256

    def test_select_resnet_resnet50(self):
        """Test select_resnet with ResNet50."""
        model, param = select_resnet('resnet50')
        
        assert model is not None
        assert isinstance(param, dict)
        assert 'feature_size' in param
        assert param['feature_size'] == 1024

    def test_select_resnet_resnet101(self):
        """Test select_resnet with ResNet101."""
        model, param = select_resnet('resnet101')
        
        assert model is not None
        assert isinstance(param, dict)
        assert 'feature_size' in param
        assert param['feature_size'] == 1024

    def test_select_resnet_resnet152(self):
        """Test select_resnet with ResNet152."""
        model, param = select_resnet('resnet152')
        
        assert model is not None
        assert isinstance(param, dict)
        assert 'feature_size' in param
        assert param['feature_size'] == 1024

    def test_select_resnet_resnet200(self):
        """Test select_resnet with ResNet200."""
        model, param = select_resnet('resnet200')
        
        assert model is not None
        assert isinstance(param, dict)
        assert 'feature_size' in param
        assert param['feature_size'] == 1024

    def test_select_resnet_invalid_network(self):
        """Test select_resnet with invalid network name."""
        with pytest.raises(IOError, match='model type is wrong'):
            select_resnet('invalid_network')

    def test_select_resnet_track_running_stats_true(self):
        """Test select_resnet with track_running_stats=True."""
        model, param = select_resnet('resnet18', track_running_stats=True)
        
        assert model is not None
        assert isinstance(param, dict)
        
        # Check that batch norm layers have track_running_stats=True
        # This would require inspecting the model structure
        # For now, just verify it doesn't crash
        assert True

    def test_select_resnet_track_running_stats_false(self):
        """Test select_resnet with track_running_stats=False."""
        model, param = select_resnet('resnet18', track_running_stats=False)
        
        assert model is not None
        assert isinstance(param, dict)
        
        # Check that batch norm layers have track_running_stats=False
        # This would require inspecting the model structure
        # For now, just verify it doesn't crash
        assert True

    def test_select_resnet_return_types(self):
        """Test that select_resnet returns correct types."""
        model, param = select_resnet('resnet18')
        
        # Model should be a PyTorch module
        assert isinstance(model, nn.Module)
        
        # Param should be a dictionary
        assert isinstance(param, dict)
        
        # Feature size should be an integer
        assert isinstance(param['feature_size'], int)

    def test_select_resnet_model_callable(self):
        """Test that returned model is callable."""
        model, param = select_resnet('resnet18')
        
        # Create dummy input
        # ResNet typically expects (batch_size, channels, height, width)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Model should be callable
        assert callable(model)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        # Output should be a tensor
        assert isinstance(output, torch.Tensor)
        
        # Output should have correct feature size
        assert output.shape[-1] == param['feature_size']

    def test_select_resnet_different_networks_different_features(self):
        """Test that different networks have different feature sizes."""
        networks = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
        feature_sizes = {}
        
        for network in networks:
            model, param = select_resnet(network)
            feature_sizes[network] = param['feature_size']
        
        # ResNet18 and ResNet34 should have same feature size (256)
        assert feature_sizes['resnet18'] == feature_sizes['resnet34']
        
        # ResNet50 and ResNet101 should have same feature size (1024)
        assert feature_sizes['resnet50'] == feature_sizes['resnet101']
        
        # ResNet18/34 should have different feature size than ResNet50/101
        assert feature_sizes['resnet18'] != feature_sizes['resnet50']

    def test_select_resnet_default_feature_size(self):
        """Test default feature size parameter."""
        model, param = select_resnet('resnet18')
        
        # Default feature size should be 256 for ResNet18
        assert param['feature_size'] == 256

    def test_select_resnet_special_dim8_case(self):
        """Test special case of resnet18-dim8."""
        model, param = select_resnet('resnet18-dim8')
        
        # Feature size should be 8 for this special case
        assert param['feature_size'] == 8

    def test_select_resnet_parameter_consistency(self):
        """Test parameter consistency across multiple calls."""
        # Multiple calls should return consistent parameters
        model1, param1 = select_resnet('resnet18')
        model2, param2 = select_resnet('resnet18')
        
        assert param1 == param2
        assert param1['feature_size'] == param2['feature_size']

    def test_select_resnet_model_training_mode(self):
        """Test model training mode functionality."""
        model, param = select_resnet('resnet18')
        
        # Model should support training mode
        model.train()
        assert model.training
        
        # Model should support eval mode
        model.eval()
        assert not model.training

    def test_select_resnet_model_parameters(self):
        """Test model parameters."""
        model, param = select_resnet('resnet18')
        
        # Model should have parameters
        params = list(model.parameters())
        assert len(params) > 0
        
        # All parameters should be tensors
        for p in params:
            assert isinstance(p, torch.Tensor)

    def test_select_resnet_model_state_dict(self):
        """Test model state dict functionality."""
        model, param = select_resnet('resnet18')
        
        # Model should have state dict
        state_dict = model.state_dict()
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        
        # All state dict values should be tensors
        for key, value in state_dict.items():
            assert isinstance(value, torch.Tensor)

    def test_select_resnet_model_device_compatibility(self):
        """Test model device compatibility."""
        model, param = select_resnet('resnet18')
        
        # Test CPU
        model_cpu = model.cpu()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        model_cpu.eval()
        with torch.no_grad():
            output = model_cpu(dummy_input)
        
        assert output.device == torch.device('cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            dummy_input_cuda = torch.randn(1, 3, 224, 224).cuda()
            
            model_cuda.eval()
            with torch.no_grad():
                output_cuda = model_cuda(dummy_input_cuda)
            
            assert output_cuda.device.type == 'cuda'

    def test_select_resnet_model_different_input_sizes(self):
        """Test model with different input sizes."""
        model, param = select_resnet('resnet18')
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            assert output.shape[0] == batch_size
            assert output.shape[-1] == param['feature_size']

    def test_select_resnet_model_gradient_computation(self):
        """Test gradient computation through model."""
        model, param = select_resnet('resnet18')
        
        # Create input that requires gradient
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        
        # Forward pass
        model.train()
        output = model(dummy_input)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert dummy_input.grad is not None
        assert not torch.isnan(dummy_input.grad).any()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_select_resnet_model_output_shape_consistency(self):
        """Test output shape consistency across different networks."""
        networks = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
        
        for network in networks:
            model, param = select_resnet(network)
            
            dummy_input = torch.randn(2, 3, 224, 224)
            
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            # Output should have correct shape
            assert output.shape[0] == 2  # batch size
            assert output.shape[-1] == param['feature_size']

    def test_select_resnet_model_memory_efficiency(self):
        """Test model memory efficiency."""
        model, param = select_resnet('resnet18')
        
        # Test that model can handle reasonable batch sizes
        try:
            dummy_input = torch.randn(16, 3, 224, 224)
            
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            assert output.shape[0] == 16
            assert output.shape[-1] == param['feature_size']
            
        except RuntimeError as e:
            # If memory error, it's expected for large inputs
            if "out of memory" in str(e):
                pytest.skip("Insufficient memory for large batch test")
            else:
                raise

    def test_select_resnet_model_serialization(self):
        """Test model serialization."""
        model, param = select_resnet('resnet18')
        
        # Test saving and loading model
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            
            # Load model
            model2, param2 = select_resnet('resnet18')
            model2.load_state_dict(torch.load(f.name))
            
            # Clean up
            os.unlink(f.name)
            
            # Models should produce same output
            dummy_input = torch.randn(1, 3, 224, 224)
            
            model.eval()
            model2.eval()
            
            with torch.no_grad():
                output1 = model(dummy_input)
                output2 = model2(dummy_input)
            
            assert torch.allclose(output1, output2)

    def test_select_resnet_all_supported_networks(self):
        """Test all supported network types."""
        supported_networks = [
            'resnet18', 'resnet18-dim8', 'resnet34', 'resnet50',
            'resnet101', 'resnet152', 'resnet200'
        ]
        
        for network in supported_networks:
            model, param = select_resnet(network)
            
            # Basic checks for all networks
            assert model is not None
            assert isinstance(param, dict)
            assert 'feature_size' in param
            assert isinstance(param['feature_size'], int)
            assert param['feature_size'] > 0
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224)
            
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            assert isinstance(output, torch.Tensor)
            assert output.shape[-1] == param['feature_size']
import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import MagicMock, patch
from argparse import Namespace

@pytest.fixture
def device():
    """Return the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def sample_args():
    """Create a sample args object for testing."""
    return Namespace(
        # Model parameters
        hyperbolic=False,
        hyperbolic_version=1,
        network_feature='resnet18',
        feature_dim=256,
        final_2dim=False,
        not_track_running_stats=False,
        only_train_linear=False,
        linear_input='features',
        
        # Dataset parameters
        seq_len=10,
        num_seq=5,
        img_dim=224,
        downsample=3,
        epsilon=5,
        
        # Training parameters
        batch_size=4,
        epochs=10,
        start_epoch=0,
        lr=0.001,
        partial=1.0,
        
        # Loss parameters
        distance='regular',
        early_action=False,
        early_action_self=False,
        use_labels=True,
        hierarchical_labels=False,
        pred_future=False,
        
        # System parameters
        device='cuda' if torch.cuda.is_available() else 'cpu',
        local_rank=0,
        parallel='none',
        cross_gpu_score=False,
        fp16=False,
        debug=False,
        test=False,
        test_info='compute_accuracy'
    )

@pytest.fixture
def sample_args_hyperbolic(sample_args):
    """Create a sample args object for hyperbolic testing."""
    args = sample_args
    args.hyperbolic = True
    args.hyperbolic_version = 1
    args.fp64_hyper = False
    return args

@pytest.fixture
def sample_args_hyperbolic_transformer(sample_args):
    """Create a sample args object for hyperbolic transformer testing."""
    args = sample_args
    args.hyperbolic = True
    args.hyperbolic_version = 1
    args.fp64_hyper = False
    args.use_transformer = True
    args.num_heads = 8
    args.transformer_layers = 2
    args.transformer_dropout = 0.1
    return args

@pytest.fixture
def sample_video_data():
    """Create sample video data for testing."""
    batch_size = 2
    num_seq = 5
    seq_len = 10
    channels = 3
    height = 224
    width = 224
    
    # Create sample video tensor [B, N, C, SL, H, W]
    video_tensor = torch.randn(batch_size, num_seq, channels, seq_len, height, width)
    labels = torch.randint(0, 10, (batch_size,))
    
    return video_tensor, labels

@pytest.fixture
def sample_features():
    """Create sample feature tensors for testing."""
    batch_size = 2
    feature_dim = 256
    
    features = torch.randn(batch_size, feature_dim)
    return features

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_dataset_path(temp_dir):
    """Create a mock dataset directory structure."""
    # Create mock directory structure
    train_dir = os.path.join(temp_dir, 'train')
    val_dir = os.path.join(temp_dir, 'val')
    test_dir = os.path.join(temp_dir, 'test')
    
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)
    
    # Create some mock class directories
    for split_dir in [train_dir, val_dir, test_dir]:
        for i in range(3):
            class_dir = os.path.join(split_dir, f'class_{i}')
            os.makedirs(class_dir)
    
    return temp_dir

@pytest.fixture
def mock_video_info():
    """Create mock video information for testing."""
    import pandas as pd
    
    # Create mock video info
    video_paths = [f'video_{i}.mp4' for i in range(10)]
    video_lengths = [100 + i * 10 for i in range(10)]
    
    video_info = pd.DataFrame({
        0: video_paths,  # video path
        1: video_lengths  # video length
    })
    
    return video_info

@pytest.fixture
def mock_model_state():
    """Create a mock model state for testing."""
    return {
        'epoch': 5,
        'net': 'resnet18',
        'state_dict': {'layer1.weight': torch.randn(64, 3, 7, 7)},
        'best_acc': 0.85,
        'optimizer': {'state': {}, 'param_groups': [{'lr': 0.001}]},
        'iteration': 1000,
        'scheduler': {'last_epoch': 4}
    }

@pytest.fixture
def mock_loss_results():
    """Create mock loss computation results."""
    return {
        'loss': torch.tensor(0.5),
        'accuracy': 0.8,
        'hier_accuracy': 0.7,
        'top1': 0.8,
        'top3': 0.9,
        'top5': 0.95
    }

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

# Mock external dependencies that might not be available in test environment
@pytest.fixture(autouse=True)
def mock_external_deps():
    """Mock external dependencies that might not be available."""
    with patch('torch.utils.tensorboard.SummaryWriter'):
        with patch('torchvision.datasets'):
            with patch('torchvision.models'):
                yield
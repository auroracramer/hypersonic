"""Tests for utils module."""

import pytest
import torch
import os
import tempfile
from unittest.mock import Mock, patch

from utils.utils import (
    AverageMeter, 
    calc_topk_accuracy, 
    calc_accuracy, 
    save_checkpoint
)


class TestAverageMeter:
    """Test cases for AverageMeter class."""

    def test_average_meter_initialization(self):
        """Test AverageMeter initialization."""
        meter = AverageMeter()
        
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0
        assert len(meter.history) == 0

    def test_average_meter_update(self):
        """Test AverageMeter update functionality."""
        meter = AverageMeter()
        
        meter.update(10, 1)
        assert meter.val == 10
        assert meter.avg == 10
        assert meter.sum == 10
        assert meter.count == 1

        meter.update(20, 2)
        assert meter.val == 20
        assert meter.avg == 16.67  # (10*1 + 20*2) / 3
        assert meter.sum == 50
        assert meter.count == 3

    def test_average_meter_reset(self):
        """Test AverageMeter reset functionality."""
        meter = AverageMeter()
        
        meter.update(10, 1)
        meter.reset()
        
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0


class TestAccuracyFunctions:
    """Test accuracy calculation functions."""

    def test_calc_accuracy_basic(self):
        """Test basic accuracy calculation."""
        # Create dummy predictions and targets
        output = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        target = torch.tensor([1, 0, 1])
        
        accuracy = calc_accuracy(output, target)
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100

    def test_calc_topk_accuracy(self):
        """Test top-k accuracy calculation."""
        # Create dummy predictions and targets
        output = torch.tensor([[0.1, 0.9, 0.05], [0.8, 0.2, 0.3], [0.3, 0.7, 0.1]])
        target = torch.tensor([1, 0, 1])
        
        top1, top2 = calc_topk_accuracy(output, target, topk=(1, 2))
        
        assert isinstance(top1, torch.Tensor)
        assert isinstance(top2, torch.Tensor)
        assert 0 <= top1.item() <= 100
        assert 0 <= top2.item() <= 100
        assert top2.item() >= top1.item()  # top-2 should be >= top-1


class TestCheckpointFunctions:
    """Test checkpoint saving functionality."""

    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, 'test_checkpoint.pth.tar')
            
            state = {
                'epoch': 1,
                'state_dict': {'param1': torch.tensor([1.0])},
                'optimizer': {'lr': 0.001},
                'best_acc': 85.5
            }
            
            save_checkpoint(state, is_best=False, filename=filename)
            
            # Check that file was created
            assert os.path.exists(filename)
            
            # Load and verify contents
            loaded_state = torch.load(filename, map_location='cpu')
            assert loaded_state['epoch'] == 1
            assert loaded_state['best_acc'] == 85.5
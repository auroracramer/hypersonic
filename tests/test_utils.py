import pytest
import torch
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime
import sys

# Add the parent directory to sys.path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import (
    save_checkpoint, write_log, calc_topk_accuracy, calc_accuracy,
    calc_accuracy_binary, denorm, AverageMeter, neq_load_customized,
    print_r
)


class TestSaveCheckpoint:
    """Test cases for the save_checkpoint function."""

    def test_save_checkpoint_basic(self, temp_dir):
        """Test basic checkpoint saving."""
        state = {
            'epoch': 10,
            'net': 'resnet18',
            'state_dict': {'layer1.weight': torch.randn(64, 3, 7, 7)},
            'best_acc': 0.85,
            'optimizer': {'state': {}, 'param_groups': [{'lr': 0.001}]},
            'iteration': 1000
        }
        
        filename = os.path.join(temp_dir, 'checkpoint.pth.tar')
        
        save_checkpoint(state, filename=filename)
        
        # Check that file was created
        assert os.path.exists(filename)
        
        # Check that state can be loaded
        loaded_state = torch.load(filename)
        assert loaded_state['epoch'] == state['epoch']
        assert loaded_state['net'] == state['net']
        assert loaded_state['best_acc'] == state['best_acc']

    def test_save_checkpoint_best(self, temp_dir):
        """Test saving best checkpoint."""
        state = {
            'epoch': 10,
            'net': 'resnet18',
            'state_dict': {'layer1.weight': torch.randn(64, 3, 7, 7)},
            'best_acc': 0.85,
            'optimizer': {'state': {}, 'param_groups': [{'lr': 0.001}]},
            'iteration': 1000
        }
        
        filename = os.path.join(temp_dir, 'checkpoint.pth.tar')
        
        save_checkpoint(state, is_best=True, filename=filename)
        
        # Check that regular checkpoint was created
        assert os.path.exists(filename)
        
        # Check that best checkpoint was created
        best_filename = os.path.join(temp_dir, 'model_best_epoch10.pth.tar')
        assert os.path.exists(best_filename)

    def test_save_checkpoint_cleanup(self, temp_dir):
        """Test checkpoint cleanup."""
        state = {
            'epoch': 10,
            'net': 'resnet18',
            'state_dict': {'layer1.weight': torch.randn(64, 3, 7, 7)},
            'best_acc': 0.85,
            'optimizer': {'state': {}, 'param_groups': [{'lr': 0.001}]},
            'iteration': 1000
        }
        
        filename = os.path.join(temp_dir, 'checkpoint.pth.tar')
        
        # Create old checkpoint
        old_filename = os.path.join(temp_dir, 'epoch9.pth.tar')
        torch.save(state, old_filename)
        
        # Save new checkpoint
        save_checkpoint(state, filename=filename, gap=1)
        
        # Check that old checkpoint was removed
        assert not os.path.exists(old_filename)

    def test_save_checkpoint_keep_all(self, temp_dir):
        """Test saving checkpoint with keep_all=True."""
        state = {
            'epoch': 10,
            'net': 'resnet18',
            'state_dict': {'layer1.weight': torch.randn(64, 3, 7, 7)},
            'best_acc': 0.85,
            'optimizer': {'state': {}, 'param_groups': [{'lr': 0.001}]},
            'iteration': 1000
        }
        
        filename = os.path.join(temp_dir, 'checkpoint.pth.tar')
        
        # Create old checkpoint
        old_filename = os.path.join(temp_dir, 'epoch9.pth.tar')
        torch.save(state, old_filename)
        
        # Save new checkpoint with keep_all=True
        save_checkpoint(state, filename=filename, gap=1, keep_all=True)
        
        # Check that old checkpoint was NOT removed
        assert os.path.exists(old_filename)

    def test_save_checkpoint_replace_best(self, temp_dir):
        """Test replacing best checkpoint."""
        state = {
            'epoch': 10,
            'net': 'resnet18',
            'state_dict': {'layer1.weight': torch.randn(64, 3, 7, 7)},
            'best_acc': 0.85,
            'optimizer': {'state': {}, 'param_groups': [{'lr': 0.001}]},
            'iteration': 1000
        }
        
        filename = os.path.join(temp_dir, 'checkpoint.pth.tar')
        
        # Create old best checkpoint
        old_best_filename = os.path.join(temp_dir, 'model_best_epoch8.pth.tar')
        torch.save(state, old_best_filename)
        
        # Save new best checkpoint
        save_checkpoint(state, is_best=True, filename=filename)
        
        # Check that old best checkpoint was removed
        assert not os.path.exists(old_best_filename)
        
        # Check that new best checkpoint was created
        new_best_filename = os.path.join(temp_dir, 'model_best_epoch10.pth.tar')
        assert os.path.exists(new_best_filename)


class TestWriteLog:
    """Test cases for the write_log function."""

    def test_write_log_new_file(self, temp_dir):
        """Test writing log to new file."""
        log_file = os.path.join(temp_dir, 'test.log')
        content = 'Test log content'
        epoch = 5
        
        write_log(content, epoch, log_file)
        
        # Check that file was created
        assert os.path.exists(log_file)
        
        # Check file content
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert f'## Epoch {epoch}:' in log_content
        assert content in log_content

    def test_write_log_append_file(self, temp_dir):
        """Test appending to existing log file."""
        log_file = os.path.join(temp_dir, 'test.log')
        
        # Write first log
        write_log('First log', 1, log_file)
        
        # Write second log
        write_log('Second log', 2, log_file)
        
        # Check that both logs are in file
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert '## Epoch 1:' in log_content
        assert '## Epoch 2:' in log_content
        assert 'First log' in log_content
        assert 'Second log' in log_content

    def test_write_log_timestamp(self, temp_dir):
        """Test that log includes timestamp."""
        log_file = os.path.join(temp_dir, 'test.log')
        content = 'Test log content'
        epoch = 5
        
        write_log(content, epoch, log_file)
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Check that timestamp is included
        assert 'time:' in log_content


class TestCalcTopkAccuracy:
    """Test cases for the calc_topk_accuracy function."""

    def test_calc_topk_accuracy_basic(self):
        """Test basic topk accuracy calculation."""
        # Create predictions and targets
        output = torch.tensor([[0.1, 0.9, 0.2, 0.3, 0.4],
                              [0.3, 0.2, 0.8, 0.1, 0.5],
                              [0.4, 0.3, 0.1, 0.9, 0.2]])
        target = torch.tensor([1, 2, 3])
        
        # Calculate top-1 accuracy
        acc = calc_topk_accuracy(output, target, topk=(1,))
        
        assert len(acc) == 1
        assert acc[0] == 100.0  # All predictions are correct

    def test_calc_topk_accuracy_top3(self):
        """Test top-3 accuracy calculation."""
        # Create predictions where top-1 fails but top-3 succeeds
        output = torch.tensor([[0.1, 0.2, 0.9, 0.3, 0.4],  # pred: 2, target: 1
                              [0.3, 0.8, 0.2, 0.1, 0.5],  # pred: 1, target: 2
                              [0.4, 0.3, 0.1, 0.9, 0.2]])  # pred: 3, target: 3
        target = torch.tensor([1, 2, 3])
        
        # Calculate top-1 and top-3 accuracy
        acc = calc_topk_accuracy(output, target, topk=(1, 3))
        
        assert len(acc) == 2
        assert acc[0] < acc[1]  # top-3 should be higher than top-1

    def test_calc_topk_accuracy_batch_size(self):
        """Test topk accuracy with different batch sizes."""
        batch_sizes = [1, 5, 10, 100]
        
        for batch_size in batch_sizes:
            output = torch.randn(batch_size, 10)
            target = torch.randint(0, 10, (batch_size,))
            
            acc = calc_topk_accuracy(output, target, topk=(1, 5))
            
            assert len(acc) == 2
            assert 0 <= acc[0] <= 100
            assert 0 <= acc[1] <= 100

    def test_calc_topk_accuracy_perfect_predictions(self):
        """Test topk accuracy with perfect predictions."""
        batch_size = 10
        num_classes = 5
        
        # Create perfect predictions
        output = torch.zeros(batch_size, num_classes)
        target = torch.randint(0, num_classes, (batch_size,))
        
        for i, t in enumerate(target):
            output[i, t] = 1.0
        
        acc = calc_topk_accuracy(output, target, topk=(1,))
        
        assert acc[0] == 100.0

    def test_calc_topk_accuracy_worst_predictions(self):
        """Test topk accuracy with worst predictions."""
        batch_size = 10
        num_classes = 5
        
        # Create worst predictions (always predict wrong class)
        output = torch.zeros(batch_size, num_classes)
        target = torch.zeros(batch_size, dtype=torch.long)
        
        for i in range(batch_size):
            output[i, 1] = 1.0  # Always predict class 1, target is class 0
        
        acc = calc_topk_accuracy(output, target, topk=(1,))
        
        assert acc[0] == 0.0


class TestCalcAccuracy:
    """Test cases for the calc_accuracy function."""

    def test_calc_accuracy_basic(self):
        """Test basic accuracy calculation."""
        output = torch.tensor([[0.1, 0.9, 0.2],
                              [0.3, 0.2, 0.8],
                              [0.9, 0.3, 0.1]])
        target = torch.tensor([1, 2, 0])
        
        acc = calc_accuracy(output, target)
        
        assert acc == 100.0  # All predictions are correct

    def test_calc_accuracy_partial(self):
        """Test partial accuracy calculation."""
        output = torch.tensor([[0.1, 0.9, 0.2],  # pred: 1, target: 1 ✓
                              [0.3, 0.2, 0.8],  # pred: 2, target: 1 ✗
                              [0.9, 0.3, 0.1]])  # pred: 0, target: 0 ✓
        target = torch.tensor([1, 1, 0])
        
        acc = calc_accuracy(output, target)
        
        assert acc == 200.0 / 3  # 2 out of 3 correct

    def test_calc_accuracy_zero(self):
        """Test zero accuracy."""
        output = torch.tensor([[0.1, 0.9, 0.2],  # pred: 1, target: 0 ✗
                              [0.3, 0.2, 0.8],  # pred: 2, target: 0 ✗
                              [0.4, 0.9, 0.1]])  # pred: 1, target: 0 ✗
        target = torch.tensor([0, 0, 0])
        
        acc = calc_accuracy(output, target)
        
        assert acc == 0.0

    def test_calc_accuracy_different_shapes(self):
        """Test accuracy with different tensor shapes."""
        batch_sizes = [1, 5, 10, 100]
        num_classes = [2, 5, 10, 100]
        
        for batch_size in batch_sizes:
            for num_class in num_classes:
                output = torch.randn(batch_size, num_class)
                target = torch.randint(0, num_class, (batch_size,))
                
                acc = calc_accuracy(output, target)
                
                assert 0 <= acc <= 100


class TestCalcAccuracyBinary:
    """Test cases for the calc_accuracy_binary function."""

    def test_calc_accuracy_binary_basic(self):
        """Test basic binary accuracy calculation."""
        output = torch.tensor([0.8, 0.2, 0.9, 0.1])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        acc = calc_accuracy_binary(output, target)
        
        assert acc == 100.0  # All predictions are correct

    def test_calc_accuracy_binary_threshold(self):
        """Test binary accuracy with threshold."""
        output = torch.tensor([0.6, 0.4, 0.7, 0.3])  # threshold is 0.5
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        acc = calc_accuracy_binary(output, target)
        
        assert acc == 100.0  # All predictions are correct

    def test_calc_accuracy_binary_partial(self):
        """Test partial binary accuracy."""
        output = torch.tensor([0.6, 0.4, 0.3, 0.7])  # pred: [1, 0, 0, 1]
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])  # target: [1, 0, 1, 0]
        
        acc = calc_accuracy_binary(output, target)
        
        assert acc == 50.0  # 2 out of 4 correct

    def test_calc_accuracy_binary_edge_cases(self):
        """Test binary accuracy edge cases."""
        # Test with exact threshold values
        output = torch.tensor([0.5, 0.5])
        target = torch.tensor([1.0, 0.0])
        
        acc = calc_accuracy_binary(output, target)
        
        assert 0 <= acc <= 100  # Should handle threshold edge case

    def test_calc_accuracy_binary_different_sizes(self):
        """Test binary accuracy with different tensor sizes."""
        sizes = [1, 5, 10, 100]
        
        for size in sizes:
            output = torch.rand(size)
            target = torch.randint(0, 2, (size,)).float()
            
            acc = calc_accuracy_binary(output, target)
            
            assert 0 <= acc <= 100


class TestDenorm:
    """Test cases for the denorm function."""

    def test_denorm_default(self):
        """Test denormalization with default parameters."""
        denorm_transform = denorm()
        
        # Test with normalized tensor
        normalized = torch.tensor([[[0.0, 1.0, -1.0]]])  # Single channel
        denormalized = denorm_transform(normalized)
        
        assert denormalized.shape == normalized.shape
        assert not torch.allclose(denormalized, normalized)

    def test_denorm_custom_params(self):
        """Test denormalization with custom parameters."""
        mean = [0.5, 0.5, 0.5]
        std = [0.2, 0.2, 0.2]
        
        denorm_transform = denorm(mean=mean, std=std)
        
        # Test with normalized tensor
        normalized = torch.zeros(1, 3, 10, 10)
        denormalized = denorm_transform(normalized)
        
        assert denormalized.shape == normalized.shape
        
        # Check that denormalization formula is applied correctly
        # denormalized = normalized * std + mean
        expected = normalized * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
        assert torch.allclose(denormalized, expected)

    def test_denorm_batch(self):
        """Test denormalization with batch of images."""
        batch_size = 4
        channels = 3
        height = 224
        width = 224
        
        denorm_transform = denorm()
        
        # Test with batch of normalized tensors
        normalized = torch.randn(batch_size, channels, height, width)
        denormalized = denorm_transform(normalized)
        
        assert denormalized.shape == normalized.shape
        assert denormalized.shape == (batch_size, channels, height, width)

    def test_denorm_single_channel(self):
        """Test denormalization with single channel."""
        mean = [0.5]
        std = [0.2]
        
        denorm_transform = denorm(mean=mean, std=std)
        
        # Test with single channel tensor
        normalized = torch.zeros(1, 1, 10, 10)
        denormalized = denorm_transform(normalized)
        
        assert denormalized.shape == normalized.shape
        expected = normalized * 0.2 + 0.5
        assert torch.allclose(denormalized, expected)


class TestAverageMeter:
    """Test cases for the AverageMeter class."""

    def test_average_meter_init(self):
        """Test AverageMeter initialization."""
        meter = AverageMeter()
        
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0

    def test_average_meter_reset(self):
        """Test AverageMeter reset."""
        meter = AverageMeter()
        
        # Update meter
        meter.update(5.0)
        
        # Reset meter
        meter.reset()
        
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0

    def test_average_meter_single_update(self):
        """Test single update to AverageMeter."""
        meter = AverageMeter()
        
        meter.update(5.0)
        
        assert meter.val == 5.0
        assert meter.avg == 5.0
        assert meter.sum == 5.0
        assert meter.count == 1

    def test_average_meter_multiple_updates(self):
        """Test multiple updates to AverageMeter."""
        meter = AverageMeter()
        
        meter.update(4.0)
        meter.update(6.0)
        
        assert meter.val == 6.0
        assert meter.avg == 5.0  # (4.0 + 6.0) / 2
        assert meter.sum == 10.0
        assert meter.count == 2

    def test_average_meter_weighted_update(self):
        """Test weighted update to AverageMeter."""
        meter = AverageMeter()
        
        meter.update(4.0, n=2)
        meter.update(6.0, n=3)
        
        assert meter.val == 6.0
        assert meter.avg == 5.2  # (4.0*2 + 6.0*3) / (2+3)
        assert meter.sum == 26.0
        assert meter.count == 5

    def test_average_meter_history(self):
        """Test AverageMeter with history."""
        meter = AverageMeter()
        
        # Update with history
        meter.update(4.0, history=1)
        meter.update(6.0, history=1)
        meter.update(8.0, history=1)
        
        # Check that history is tracked
        assert len(meter.history) == 3
        assert meter.history[-1] == 8.0

    def test_average_meter_dict_update(self):
        """Test AverageMeter dict_update method."""
        meter = AverageMeter()
        
        values = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        
        meter.dict_update(values, 'b')
        
        assert meter.val == 2.0
        assert meter.avg == 2.0
        assert meter.sum == 2.0
        assert meter.count == 1

    def test_average_meter_len(self):
        """Test AverageMeter __len__ method."""
        meter = AverageMeter()
        
        assert len(meter) == 0
        
        meter.update(5.0)
        assert len(meter) == 1
        
        meter.update(3.0)
        assert len(meter) == 2

    def test_average_meter_step_history(self):
        """Test AverageMeter with step history."""
        meter = AverageMeter()
        
        # Update with step history
        for i in range(10):
            meter.update(float(i), history=1, step=3)
        
        # Check that step history is working
        assert len(meter.history) <= 10


class TestNeqLoadCustomized:
    """Test cases for the neq_load_customized function."""

    def test_neq_load_customized_basic(self, sample_args):
        """Test basic customized loading."""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Create pretrained dict
        pretrained_dict = {
            '0.weight': torch.randn(5, 10),
            '0.bias': torch.randn(5),
            '2.weight': torch.randn(2, 5),
            '2.bias': torch.randn(2)
        }
        
        # Load customized
        neq_load_customized(sample_args, model, pretrained_dict)
        
        # Check that weights were loaded
        assert torch.allclose(model[0].weight, pretrained_dict['0.weight'])
        assert torch.allclose(model[0].bias, pretrained_dict['0.bias'])

    def test_neq_load_customized_size_mismatch(self, sample_args):
        """Test customized loading with size mismatch."""
        model = torch.nn.Linear(10, 5)
        
        # Create pretrained dict with different size
        pretrained_dict = {
            'weight': torch.randn(3, 8),  # Different size
            'bias': torch.randn(3)
        }
        
        # Should not crash, should skip mismatched layers
        neq_load_customized(sample_args, model, pretrained_dict, size_diff=True)

    def test_neq_load_customized_partial_match(self, sample_args):
        """Test customized loading with partial match."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Create pretrained dict with only some layers
        pretrained_dict = {
            '0.weight': torch.randn(5, 10),
            '0.bias': torch.randn(5)
            # Missing layer 2
        }
        
        # Should load what matches
        neq_load_customized(sample_args, model, pretrained_dict)
        
        # Check that matched weights were loaded
        assert torch.allclose(model[0].weight, pretrained_dict['0.weight'])
        assert torch.allclose(model[0].bias, pretrained_dict['0.bias'])

    def test_neq_load_customized_different_parts(self, sample_args):
        """Test customized loading with different parts."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        pretrained_dict = {
            '0.weight': torch.randn(5, 10),
            '0.bias': torch.randn(5),
            '2.weight': torch.randn(2, 5),
            '2.bias': torch.randn(2)
        }
        
        # Test loading specific parts
        parts = ['backbone', 'agg']
        neq_load_customized(sample_args, model, pretrained_dict, parts=parts)


class TestPrintR:
    """Test cases for the print_r function."""

    def test_print_r_basic(self, sample_args, capsys):
        """Test basic print_r functionality."""
        text = "Test message"
        
        print_r(sample_args, text)
        
        captured = capsys.readouterr()
        assert text in captured.out

    def test_print_r_with_local_rank(self, sample_args, capsys):
        """Test print_r with local_rank."""
        sample_args.local_rank = 1
        text = "Test message"
        
        print_r(sample_args, text)
        
        captured = capsys.readouterr()
        assert captured.out == ""  # Should not print for local_rank > 0

    def test_print_r_print_no_verbose(self, sample_args, capsys):
        """Test print_r with print_no_verbose flag."""
        sample_args.local_rank = 1
        text = "Test message"
        
        print_r(sample_args, text, print_no_verbose=True)
        
        captured = capsys.readouterr()
        assert text in captured.out  # Should print regardless of local_rank

    def test_print_r_different_ranks(self, sample_args, capsys):
        """Test print_r with different local ranks."""
        text = "Test message"
        
        # Test with rank 0
        sample_args.local_rank = 0
        print_r(sample_args, text)
        captured = capsys.readouterr()
        assert text in captured.out
        
        # Test with rank -1
        sample_args.local_rank = -1
        print_r(sample_args, text)
        captured = capsys.readouterr()
        assert text in captured.out
        
        # Test with rank 2
        sample_args.local_rank = 2
        print_r(sample_args, text)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    def test_checkpoint_workflow(self, temp_dir):
        """Test complete checkpoint workflow."""
        # Create initial state
        state = {
            'epoch': 1,
            'net': 'resnet18',
            'state_dict': {'layer1.weight': torch.randn(64, 3, 7, 7)},
            'best_acc': 0.5,
            'optimizer': {'state': {}, 'param_groups': [{'lr': 0.001}]},
            'iteration': 100
        }
        
        filename = os.path.join(temp_dir, 'checkpoint.pth.tar')
        
        # Save initial checkpoint
        save_checkpoint(state, filename=filename)
        
        # Update state and save as best
        state['epoch'] = 2
        state['best_acc'] = 0.8
        save_checkpoint(state, is_best=True, filename=filename)
        
        # Load and verify
        loaded_state = torch.load(filename)
        assert loaded_state['epoch'] == 2
        assert loaded_state['best_acc'] == 0.8
        
        # Check best checkpoint exists
        best_filename = os.path.join(temp_dir, 'model_best_epoch2.pth.tar')
        assert os.path.exists(best_filename)

    def test_accuracy_consistency(self):
        """Test consistency between different accuracy functions."""
        # Test that topk(1) matches regular accuracy
        output = torch.randn(10, 5)
        target = torch.randint(0, 5, (10,))
        
        acc1 = calc_accuracy(output, target)
        acc_topk = calc_topk_accuracy(output, target, topk=(1,))[0]
        
        assert abs(acc1 - acc_topk) < 1e-5  # Should be very close

    def test_meter_logging_workflow(self, temp_dir):
        """Test workflow combining meters and logging."""
        # Create meters
        meters = {
            'loss': AverageMeter(),
            'acc': AverageMeter()
        }
        
        # Simulate training updates
        for i in range(5):
            meters['loss'].update(1.0 / (i + 1))
            meters['acc'].update(i * 20.0)
        
        # Log results
        log_file = os.path.join(temp_dir, 'train.log')
        content = f"Loss: {meters['loss'].avg:.4f}, Acc: {meters['acc'].avg:.2f}"
        write_log(content, 1, log_file)
        
        # Verify log
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "Loss:" in log_content
        assert "Acc:" in log_content
        assert "## Epoch 1:" in log_content
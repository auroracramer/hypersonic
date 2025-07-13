import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import os
import tempfile
from unittest.mock import patch, MagicMock, call
from collections import OrderedDict
import sys

# Add the parent directory to sys.path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import Trainer, gather_tensor
from utils.utils import AverageMeter


class TestTrainer:
    """Test cases for the Trainer class."""

    def test_trainer_init(self, sample_args):
        """Test Trainer initialization."""
        # Create mock components
        model = MagicMock()
        optimizer = MagicMock()
        loaders = {'train': MagicMock(), 'val': MagicMock(), 'test': MagicMock()}
        iteration = 1000
        best_acc = 0.85
        writer_train = MagicMock()
        writer_val = MagicMock()
        img_path = '/path/to/images'
        model_path = '/path/to/models'
        scheduler = MagicMock()

        trainer = Trainer(
            sample_args, model, optimizer, loaders, iteration, best_acc,
            writer_train, writer_val, img_path, model_path, scheduler
        )

        assert trainer.args == sample_args
        assert trainer.model == model
        assert trainer.optimizer == optimizer
        assert trainer.loaders == loaders
        assert trainer.iteration == iteration
        assert trainer.best_acc == best_acc
        assert trainer.writers['train'] == writer_train
        assert trainer.writers['val'] == writer_val
        assert trainer.img_path == img_path
        assert trainer.model_path == model_path
        assert trainer.scheduler == scheduler
        assert trainer.target is None
        assert trainer.sizes_mask is None

    def test_trainer_init_with_scaler(self, sample_args):
        """Test Trainer initialization includes GradScaler."""
        model = MagicMock()
        optimizer = MagicMock()
        loaders = {'train': MagicMock(), 'val': MagicMock(), 'test': MagicMock()}
        iteration = 1000
        best_acc = 0.85
        writer_train = MagicMock()
        writer_val = MagicMock()
        img_path = '/path/to/images'
        model_path = '/path/to/models'
        scheduler = MagicMock()

        trainer = Trainer(
            sample_args, model, optimizer, loaders, iteration, best_acc,
            writer_train, writer_val, img_path, model_path, scheduler
        )

        assert hasattr(trainer, 'scaler')
        assert trainer.scaler is not None

    @patch('trainer.save_checkpoint')
    def test_trainer_train_basic(self, mock_save_checkpoint, sample_args, temp_dir):
        """Test basic training loop."""
        # Setup
        sample_args.start_epoch = 0
        sample_args.epochs = 2
        sample_args.local_rank = 0
        sample_args.debug = False
        
        model = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        # Mock data loader
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter([
            (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,)))
        ]))
        mock_loader.__len__ = MagicMock(return_value=1)
        
        loaders = {'train': mock_loader, 'val': mock_loader, 'test': mock_loader}
        
        model_path = temp_dir
        
        trainer = Trainer(
            sample_args, model, optimizer, loaders, 1000, 0.5,
            MagicMock(), MagicMock(), '/img/path', model_path, scheduler
        )
        
        # Mock run_epoch to return accuracy
        trainer.run_epoch = MagicMock(side_effect=[None, 0.8, None, 0.9])
        
        # Run training
        trainer.train()
        
        # Verify run_epoch was called correctly
        expected_calls = [
            call(0, train=True),
            call(0, train=False),
            call(1, train=True),
            call(1, train=False)
        ]
        trainer.run_epoch.assert_has_calls(expected_calls)
        
        # Verify checkpoint saving
        assert mock_save_checkpoint.call_count == 2

    @patch('trainer.save_checkpoint')
    def test_trainer_train_best_accuracy_update(self, mock_save_checkpoint, sample_args, temp_dir):
        """Test training loop updates best accuracy."""
        sample_args.start_epoch = 0
        sample_args.epochs = 1
        sample_args.local_rank = 0
        sample_args.debug = False
        
        model = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        loaders = {'train': MagicMock(), 'val': MagicMock(), 'test': MagicMock()}
        
        trainer = Trainer(
            sample_args, model, optimizer, loaders, 1000, 0.5,
            MagicMock(), MagicMock(), '/img/path', temp_dir, scheduler
        )
        
        # Mock run_epoch to return improving accuracy
        trainer.run_epoch = MagicMock(side_effect=[None, 0.8])
        
        # Run training
        trainer.train()
        
        # Verify best_acc was updated
        assert trainer.best_acc == 0.8

    def test_trainer_test_compute_accuracy(self, sample_args):
        """Test test method with compute_accuracy."""
        sample_args.test_info = 'compute_accuracy'
        sample_args.local_rank = 0
        
        model = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        loaders = {'train': MagicMock(), 'val': MagicMock(), 'test': MagicMock()}
        
        trainer = Trainer(
            sample_args, model, optimizer, loaders, 1000, 0.5,
            MagicMock(), MagicMock(), '/img/path', '/model/path', scheduler
        )
        
        # Mock run_epoch to return test accuracy
        trainer.run_epoch = MagicMock(return_value=0.85)
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            trainer.test()
        
        # Verify run_epoch was called correctly
        trainer.run_epoch.assert_called_once_with(epoch=None, train=False)
        
        # Verify print was called
        mock_print.assert_called()

    def test_trainer_test_not_implemented(self, sample_args):
        """Test test method with not implemented test_info."""
        sample_args.test_info = 'not_implemented'
        sample_args.local_rank = 0
        
        model = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        loaders = {'train': MagicMock(), 'val': MagicMock(), 'test': MagicMock()}
        
        trainer = Trainer(
            sample_args, model, optimizer, loaders, 1000, 0.5,
            MagicMock(), MagicMock(), '/img/path', '/model/path', scheduler
        )
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            trainer.test()
        
        # Verify correct message was printed
        mock_print.assert_called_with('Test not_implemented is not implemented')

    @patch('trainer.time.time')
    def test_trainer_run_epoch_train(self, mock_time, sample_args):
        """Test run_epoch in training mode."""
        sample_args.device = 'cpu'
        sample_args.local_rank = 0
        sample_args.partial = 1.0
        sample_args.fp16 = False
        sample_args.cross_gpu_score = False
        
        # Setup mock model
        model = MagicMock()
        model.train = MagicMock()
        model.eval = MagicMock()
        
        # Mock model forward pass
        mock_output = (torch.randn(2, 10), torch.randn(2, 2), torch.tensor([2, 2]))
        model.return_value = mock_output
        
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        # Mock data loader
        mock_data = [
            (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,))),
            (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,)))
        ]
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter(mock_data))
        mock_loader.__len__ = MagicMock(return_value=len(mock_data))
        
        loaders = {'train': mock_loader, 'val': mock_loader, 'test': mock_loader}
        
        # Mock time
        mock_time.side_effect = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        trainer = Trainer(
            sample_args, model, optimizer, loaders, 1000, 0.5,
            MagicMock(), MagicMock(), '/img/path', '/model/path', scheduler
        )
        
        # Mock losses.compute_loss
        with patch('trainer.losses.compute_loss') as mock_compute_loss:
            mock_compute_loss.return_value = [
                torch.tensor(0.5),  # loss
                torch.tensor(0.8),  # accuracy
                torch.tensor(0.7),  # hier_accuracy
                torch.tensor(0.8),  # top1
                torch.tensor(0.9),  # top3
                torch.tensor(0.95), # top5
                torch.tensor(0.85), # pos_acc
                torch.tensor(0.15)  # neg_acc
            ]
            
            # Mock losses.bookkeeping
            with patch('trainer.losses.bookkeeping') as mock_bookkeeping:
                # Run epoch
                result = trainer.run_epoch(epoch=1, train=True)
                
                # Verify model was set to training mode
                model.train.assert_called_once()
                
                # Verify optimizer was called
                assert optimizer.zero_grad.call_count == len(mock_data)
                assert optimizer.step.call_count == len(mock_data)
                
                # Verify compute_loss was called
                assert mock_compute_loss.call_count == len(mock_data)
                
                # Verify bookkeeping was called
                assert mock_bookkeeping.call_count == len(mock_data)

    @patch('trainer.time.time')
    def test_trainer_run_epoch_eval(self, mock_time, sample_args):
        """Test run_epoch in evaluation mode."""
        sample_args.device = 'cpu'
        sample_args.local_rank = 0
        sample_args.partial = 1.0
        sample_args.fp16 = False
        sample_args.cross_gpu_score = False
        
        # Setup mock model
        model = MagicMock()
        model.train = MagicMock()
        model.eval = MagicMock()
        
        # Mock model forward pass
        mock_output = (torch.randn(2, 10), torch.randn(2, 2), torch.tensor([2, 2]))
        model.return_value = mock_output
        
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        # Mock data loader
        mock_data = [
            (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,)))
        ]
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter(mock_data))
        mock_loader.__len__ = MagicMock(return_value=len(mock_data))
        
        loaders = {'train': mock_loader, 'val': mock_loader, 'test': mock_loader}
        
        # Mock time
        mock_time.side_effect = [0.0, 0.1, 0.2, 0.3]
        
        trainer = Trainer(
            sample_args, model, optimizer, loaders, 1000, 0.5,
            MagicMock(), MagicMock(), '/img/path', '/model/path', scheduler
        )
        
        # Mock losses.compute_loss
        with patch('trainer.losses.compute_loss') as mock_compute_loss:
            mock_compute_loss.return_value = [
                torch.tensor(0.5),  # loss
                torch.tensor(0.8),  # accuracy
                torch.tensor(0.7),  # hier_accuracy
                torch.tensor(0.8),  # top1
                torch.tensor(0.9),  # top3
                torch.tensor(0.95), # top5
                torch.tensor(0.85), # pos_acc
                torch.tensor(0.15)  # neg_acc
            ]
            
            # Mock losses.bookkeeping
            with patch('trainer.losses.bookkeeping') as mock_bookkeeping:
                # Run epoch
                result = trainer.run_epoch(epoch=1, train=False)
                
                # Verify model was set to evaluation mode
                model.eval.assert_called_once()
                
                # Verify optimizer was NOT called (eval mode)
                optimizer.zero_grad.assert_not_called()
                optimizer.step.assert_not_called()
                
                # Verify compute_loss was called
                assert mock_compute_loss.call_count == len(mock_data)
                
                # Verify bookkeeping was called
                assert mock_bookkeeping.call_count == len(mock_data)

    def test_trainer_run_epoch_test_mode(self, sample_args):
        """Test run_epoch in test mode (epoch=None)."""
        sample_args.device = 'cpu'
        sample_args.local_rank = 0
        sample_args.partial = 1.0
        sample_args.fp16 = False
        sample_args.cross_gpu_score = False
        
        # Setup mock model
        model = MagicMock()
        model.eval = MagicMock()
        
        # Mock model forward pass
        mock_output = (torch.randn(2, 10), torch.randn(2, 2), torch.tensor([2, 2]))
        model.return_value = mock_output
        
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        # Mock data loader
        mock_data = [
            (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,)))
        ]
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter(mock_data))
        mock_loader.__len__ = MagicMock(return_value=len(mock_data))
        
        loaders = {'train': mock_loader, 'val': mock_loader, 'test': mock_loader}
        
        trainer = Trainer(
            sample_args, model, optimizer, loaders, 1000, 0.5,
            MagicMock(), MagicMock(), '/img/path', '/model/path', scheduler
        )
        
        # Mock losses.compute_loss
        with patch('trainer.losses.compute_loss') as mock_compute_loss:
            mock_compute_loss.return_value = [
                torch.tensor(0.5),  # loss
                torch.tensor(0.8),  # accuracy
                torch.tensor(0.7),  # hier_accuracy
                torch.tensor(0.8),  # top1
                torch.tensor(0.9),  # top3
                torch.tensor(0.95), # top5
                torch.tensor(0.85), # pos_acc
                torch.tensor(0.15)  # neg_acc
            ]
            
            # Mock losses.bookkeeping
            with patch('trainer.losses.bookkeeping') as mock_bookkeeping:
                # Run epoch in test mode
                result = trainer.run_epoch(epoch=None, train=False)
                
                # Verify model was set to evaluation mode
                model.eval.assert_called_once()
                
                # Verify test loader was used
                # (This depends on implementation details)

    def test_trainer_run_epoch_with_fp16(self, sample_args):
        """Test run_epoch with fp16 enabled."""
        sample_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sample_args.local_rank = 0
        sample_args.partial = 1.0
        sample_args.fp16 = True
        sample_args.cross_gpu_score = False
        
        # Setup mock model
        model = MagicMock()
        model.train = MagicMock()
        
        # Mock model forward pass
        mock_output = (torch.randn(2, 10), torch.randn(2, 2), torch.tensor([2, 2]))
        model.return_value = mock_output
        
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        # Mock data loader
        mock_data = [
            (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,)))
        ]
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter(mock_data))
        mock_loader.__len__ = MagicMock(return_value=len(mock_data))
        
        loaders = {'train': mock_loader, 'val': mock_loader, 'test': mock_loader}
        
        trainer = Trainer(
            sample_args, model, optimizer, loaders, 1000, 0.5,
            MagicMock(), MagicMock(), '/img/path', '/model/path', scheduler
        )
        
        # Mock losses.compute_loss
        with patch('trainer.losses.compute_loss') as mock_compute_loss:
            mock_compute_loss.return_value = [
                torch.tensor(0.5),  # loss
                torch.tensor(0.8),  # accuracy
                torch.tensor(0.7),  # hier_accuracy
                torch.tensor(0.8),  # top1
                torch.tensor(0.9),  # top3
                torch.tensor(0.95), # top5
                torch.tensor(0.85), # pos_acc
                torch.tensor(0.15)  # neg_acc
            ]
            
            # Mock losses.bookkeeping
            with patch('trainer.losses.bookkeeping') as mock_bookkeeping:
                # Mock scaler
                trainer.scaler = MagicMock()
                
                # Run epoch
                result = trainer.run_epoch(epoch=1, train=True)
                
                # Verify scaler was used
                trainer.scaler.scale.assert_called()
                trainer.scaler.step.assert_called()
                trainer.scaler.update.assert_called()

    def test_trainer_run_epoch_partial_training(self, sample_args):
        """Test run_epoch with partial training."""
        sample_args.device = 'cpu'
        sample_args.local_rank = 0
        sample_args.partial = 0.5  # Only process 50% of data
        sample_args.fp16 = False
        sample_args.cross_gpu_score = False
        
        # Setup mock model
        model = MagicMock()
        model.train = MagicMock()
        
        # Mock model forward pass
        mock_output = (torch.randn(2, 10), torch.randn(2, 2), torch.tensor([2, 2]))
        model.return_value = mock_output
        
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        # Mock data loader with 4 items
        mock_data = [
            (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,))),
            (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,))),
            (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,))),
            (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,)))
        ]
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter(mock_data))
        mock_loader.__len__ = MagicMock(return_value=len(mock_data))
        
        loaders = {'train': mock_loader, 'val': mock_loader, 'test': mock_loader}
        
        trainer = Trainer(
            sample_args, model, optimizer, loaders, 1000, 0.5,
            MagicMock(), MagicMock(), '/img/path', '/model/path', scheduler
        )
        
        # Mock losses.compute_loss
        with patch('trainer.losses.compute_loss') as mock_compute_loss:
            mock_compute_loss.return_value = [
                torch.tensor(0.5),  # loss
                torch.tensor(0.8),  # accuracy
                torch.tensor(0.7),  # hier_accuracy
                torch.tensor(0.8),  # top1
                torch.tensor(0.9),  # top3
                torch.tensor(0.95), # top5
                torch.tensor(0.85), # pos_acc
                torch.tensor(0.15)  # neg_acc
            ]
            
            # Mock losses.bookkeeping
            with patch('trainer.losses.bookkeeping') as mock_bookkeeping:
                # Run epoch
                result = trainer.run_epoch(epoch=1, train=True)
                
                # Verify only partial data was processed
                # With partial=0.5 and 4 items, should process 2 items
                assert mock_compute_loss.call_count == 2

    def test_trainer_get_base_model(self, sample_args):
        """Test get_base_model method."""
        # Test with model that has 'module' attribute (DataParallel)
        base_model = MagicMock()
        model_with_module = MagicMock()
        model_with_module.module = base_model
        
        trainer = Trainer(
            sample_args, model_with_module, MagicMock(), {}, 1000, 0.5,
            MagicMock(), MagicMock(), '/img/path', '/model/path', MagicMock()
        )
        
        result = trainer.get_base_model()
        assert result == base_model
        
        # Test with model that doesn't have 'module' attribute
        model_without_module = MagicMock()
        del model_without_module.module  # Remove module attribute
        
        trainer.model = model_without_module
        result = trainer.get_base_model()
        assert result == model_without_module

    def test_trainer_cuda_sync(self, sample_args):
        """Test CUDA synchronization in run_epoch."""
        sample_args.device = 'cuda'
        sample_args.local_rank = 0
        sample_args.partial = 1.0
        sample_args.fp16 = False
        sample_args.cross_gpu_score = False
        
        model = MagicMock()
        model.eval = MagicMock()
        
        # Mock model forward pass
        mock_output = (torch.randn(2, 10), torch.randn(2, 2), torch.tensor([2, 2]))
        model.return_value = mock_output
        
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        # Mock data loader
        mock_data = [
            (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,)))
        ]
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter(mock_data))
        mock_loader.__len__ = MagicMock(return_value=len(mock_data))
        
        loaders = {'train': mock_loader, 'val': mock_loader, 'test': mock_loader}
        
        trainer = Trainer(
            sample_args, model, optimizer, loaders, 1000, 0.5,
            MagicMock(), MagicMock(), '/img/path', '/model/path', scheduler
        )
        
        # Mock CUDA synchronization
        with patch('torch.cuda.synchronize') as mock_sync:
            with patch('trainer.losses.compute_loss') as mock_compute_loss:
                mock_compute_loss.return_value = [
                    torch.tensor(0.5),  # loss
                    torch.tensor(0.8),  # accuracy
                    torch.tensor(0.7),  # hier_accuracy
                    torch.tensor(0.8),  # top1
                    torch.tensor(0.9),  # top3
                    torch.tensor(0.95), # top5
                    torch.tensor(0.85), # pos_acc
                    torch.tensor(0.15)  # neg_acc
                ]
                
                with patch('trainer.losses.bookkeeping') as mock_bookkeeping:
                    # Run epoch
                    trainer.run_epoch(epoch=1, train=False)
                    
                    # Verify CUDA sync was called
                    mock_sync.assert_called()


class TestGatherTensor:
    """Test cases for the gather_tensor function."""

    def test_gather_tensor_basic(self):
        """Test basic gather_tensor functionality."""
        # Create a simple tensor
        tensor = torch.randn(2, 3)
        
        # Mock torch.distributed functions
        with patch('torch.distributed.get_world_size') as mock_world_size:
            with patch('torch.distributed.all_gather') as mock_all_gather:
                mock_world_size.return_value = 2
                
                # Mock all_gather to simulate gathering from 2 processes
                def mock_gather_fn(tensor_list, tensor):
                    tensor_list[0] = tensor
                    tensor_list[1] = tensor
                
                mock_all_gather.side_effect = mock_gather_fn
                
                # Test gather_tensor
                result = gather_tensor(tensor)
                
                # Verify result shape (should be concatenated)
                assert result.shape[0] == tensor.shape[0] * 2
                assert result.shape[1:] == tensor.shape[1:]

    def test_gather_tensor_single_process(self):
        """Test gather_tensor with single process."""
        tensor = torch.randn(2, 3)
        
        # Mock world_size = 1 (single process)
        with patch('torch.distributed.get_world_size') as mock_world_size:
            mock_world_size.return_value = 1
            
            # Test gather_tensor
            result = gather_tensor(tensor)
            
            # Result should be the same as input for single process
            assert torch.equal(result, tensor)

    def test_gather_tensor_different_shapes(self):
        """Test gather_tensor with different tensor shapes."""
        shapes = [(1, 10), (4, 5, 3), (2, 2, 2, 2)]
        
        for shape in shapes:
            tensor = torch.randn(*shape)
            
            with patch('torch.distributed.get_world_size') as mock_world_size:
                with patch('torch.distributed.all_gather') as mock_all_gather:
                    mock_world_size.return_value = 3
                    
                    def mock_gather_fn(tensor_list, tensor):
                        for i in range(len(tensor_list)):
                            tensor_list[i] = tensor
                    
                    mock_all_gather.side_effect = mock_gather_fn
                    
                    result = gather_tensor(tensor)
                    
                    # Verify first dimension is multiplied by world_size
                    assert result.shape[0] == tensor.shape[0] * 3
                    assert result.shape[1:] == tensor.shape[1:]

    def test_gather_tensor_empty_tensor(self):
        """Test gather_tensor with empty tensor."""
        tensor = torch.empty(0, 3)
        
        with patch('torch.distributed.get_world_size') as mock_world_size:
            with patch('torch.distributed.all_gather') as mock_all_gather:
                mock_world_size.return_value = 2
                
                def mock_gather_fn(tensor_list, tensor):
                    tensor_list[0] = tensor
                    tensor_list[1] = tensor
                
                mock_all_gather.side_effect = mock_gather_fn
                
                result = gather_tensor(tensor)
                
                # Even empty tensors should be handled
                assert result.shape[0] == 0
                assert result.shape[1:] == tensor.shape[1:]


class TestTrainerIntegration:
    """Integration tests for the Trainer class."""

    def test_trainer_complete_workflow(self, sample_args, temp_dir):
        """Test complete training workflow."""
        sample_args.start_epoch = 0
        sample_args.epochs = 1
        sample_args.local_rank = 0
        sample_args.debug = False
        sample_args.device = 'cpu'
        sample_args.partial = 1.0
        sample_args.fp16 = False
        sample_args.cross_gpu_score = False
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        # Create simple data loader
        class SimpleDataset:
            def __init__(self, size=10):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return torch.randn(5, 3, 10, 224, 224), torch.randint(0, 2, (1,)).squeeze()
        
        from torch.utils.data import DataLoader
        dataset = SimpleDataset()
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        loaders = {'train': loader, 'val': loader, 'test': loader}
        
        # Create trainer
        trainer = Trainer(
            sample_args, model, optimizer, loaders, 0, 0.0,
            MagicMock(), MagicMock(), temp_dir, temp_dir, scheduler
        )
        
        # Mock losses.compute_loss and losses.bookkeeping
        with patch('trainer.losses.compute_loss') as mock_compute_loss:
            with patch('trainer.losses.bookkeeping') as mock_bookkeeping:
                mock_compute_loss.return_value = [
                    torch.tensor(0.5),  # loss
                    torch.tensor(0.8),  # accuracy
                    torch.tensor(0.7),  # hier_accuracy
                    torch.tensor(0.8),  # top1
                    torch.tensor(0.9),  # top3
                    torch.tensor(0.95), # top5
                    torch.tensor(0.85), # pos_acc
                    torch.tensor(0.15)  # neg_acc
                ]
                
                # Mock save_checkpoint
                with patch('trainer.save_checkpoint') as mock_save:
                    # Run training
                    trainer.train()
                    
                    # Verify training completed
                    assert mock_compute_loss.call_count > 0
                    assert mock_bookkeeping.call_count > 0
                    assert mock_save.call_count > 0

    def test_trainer_with_different_loaders(self, sample_args):
        """Test trainer with different loader configurations."""
        sample_args.device = 'cpu'
        sample_args.local_rank = 0
        sample_args.partial = 1.0
        sample_args.fp16 = False
        sample_args.cross_gpu_score = False
        
        model = MagicMock()
        model.eval = MagicMock()
        
        # Mock model forward pass
        mock_output = (torch.randn(2, 10), torch.randn(2, 2), torch.tensor([2, 2]))
        model.return_value = mock_output
        
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        # Test with different loader setups
        loader_configs = [
            {'train': MagicMock(), 'val': MagicMock(), 'test': MagicMock()},
            {'train': MagicMock(), 'val': MagicMock()},  # No test loader
            {'train': MagicMock(), 'test': MagicMock()},  # No val loader
        ]
        
        for loaders in loader_configs:
            # Setup mock data
            for loader in loaders.values():
                mock_data = [
                    (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,)))
                ]
                loader.__iter__ = MagicMock(return_value=iter(mock_data))
                loader.__len__ = MagicMock(return_value=len(mock_data))
            
            trainer = Trainer(
                sample_args, model, optimizer, loaders, 1000, 0.5,
                MagicMock(), MagicMock(), '/img/path', '/model/path', scheduler
            )
            
            # Mock losses.compute_loss
            with patch('trainer.losses.compute_loss') as mock_compute_loss:
                mock_compute_loss.return_value = [
                    torch.tensor(0.5),  # loss
                    torch.tensor(0.8),  # accuracy
                    torch.tensor(0.7),  # hier_accuracy
                    torch.tensor(0.8),  # top1
                    torch.tensor(0.9),  # top3
                    torch.tensor(0.95), # top5
                    torch.tensor(0.85), # pos_acc
                    torch.tensor(0.15)  # neg_acc
                ]
                
                with patch('trainer.losses.bookkeeping') as mock_bookkeeping:
                    # Test different modes
                    if 'train' in loaders:
                        trainer.run_epoch(epoch=1, train=True)
                    if 'val' in loaders:
                        trainer.run_epoch(epoch=1, train=False)
                    if 'test' in loaders:
                        trainer.run_epoch(epoch=None, train=False)
                    
                    # Should not crash
                    assert True

    def test_trainer_error_handling(self, sample_args):
        """Test trainer error handling."""
        sample_args.device = 'cpu'
        sample_args.local_rank = 0
        sample_args.partial = 1.0
        sample_args.fp16 = False
        sample_args.cross_gpu_score = False
        
        model = MagicMock()
        model.eval = MagicMock()
        
        # Mock model to raise exception
        model.side_effect = RuntimeError("Model error")
        
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        # Mock data loader
        mock_data = [
            (torch.randn(2, 5, 3, 10, 224, 224), torch.randint(0, 10, (2,)))
        ]
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter(mock_data))
        mock_loader.__len__ = MagicMock(return_value=len(mock_data))
        
        loaders = {'train': mock_loader, 'val': mock_loader, 'test': mock_loader}
        
        trainer = Trainer(
            sample_args, model, optimizer, loaders, 1000, 0.5,
            MagicMock(), MagicMock(), '/img/path', '/model/path', scheduler
        )
        
        # Test that exception is properly handled (or propagated)
        with pytest.raises(RuntimeError):
            trainer.run_epoch(epoch=1, train=False)
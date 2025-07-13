import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import losses
from utils.utils import AverageMeter


class TestComputeLoss:
    """Test cases for the compute_loss function."""

    def test_compute_loss_supervised(self, sample_args):
        """Test compute_loss with supervised learning."""
        sample_args.use_labels = True
        
        # Create mock data
        feature_dist = torch.randn(4, 4)
        pred = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        target = torch.randn(4, 256)
        sizes_pred = torch.tensor([4, 4, 4, 4])
        sizes_mask = torch.ones(4, 4)
        B = 4
        
        results = losses.compute_loss(
            sample_args, feature_dist, pred, labels, target, sizes_pred, sizes_mask, B
        )
        
        assert len(results) == 7  # loss + 6 metrics
        assert isinstance(results[0], torch.Tensor)  # loss
        assert results[0].item() >= 0  # loss should be non-negative

    def test_compute_loss_selfsupervised(self, sample_args):
        """Test compute_loss with self-supervised learning."""
        sample_args.use_labels = False
        
        # Create mock data
        feature_dist = torch.randn(4, 4)
        pred = torch.randn(4, 256)
        labels = torch.randint(0, 10, (4,))
        target = torch.randn(4, 256)
        sizes_pred = torch.tensor([4, 4, 4, 4])
        sizes_mask = torch.ones(4, 4)
        B = 4
        
        results = losses.compute_loss(
            sample_args, feature_dist, pred, labels, target, sizes_pred, sizes_mask, B
        )
        
        assert len(results) == 7  # loss + 6 metrics
        assert isinstance(results[0], torch.Tensor)  # loss
        assert results[0].item() >= 0  # loss should be non-negative

    def test_compute_loss_different_batch_sizes(self, sample_args):
        """Test compute_loss with different batch sizes."""
        sample_args.use_labels = True
        
        batch_sizes = [1, 2, 4, 8]
        
        for B in batch_sizes:
            feature_dist = torch.randn(B, B)
            pred = torch.randn(B, 10)
            labels = torch.randint(0, 10, (B,))
            target = torch.randn(B, 256)
            sizes_pred = torch.tensor([B] * B)
            sizes_mask = torch.ones(B, B)
            
            results = losses.compute_loss(
                sample_args, feature_dist, pred, labels, target, sizes_pred, sizes_mask, B
            )
            
            assert len(results) == 7
            assert isinstance(results[0], torch.Tensor)
            assert results[0].item() >= 0


class TestComputeSupervisedLoss:
    """Test cases for the compute_supervised_loss function."""

    def test_compute_supervised_loss_basic(self, sample_args):
        """Test basic supervised loss computation."""
        sample_args.hierarchical_labels = False
        sample_args.pred_future = False
        
        pred = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        B = 4
        
        results, loss = losses.compute_supervised_loss(sample_args, pred, labels, B)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert len(results) == 6  # 6 metrics returned

    def test_compute_supervised_loss_hierarchical(self, sample_args):
        """Test supervised loss with hierarchical labels."""
        sample_args.hierarchical_labels = True
        sample_args.pred_future = False
        
        pred = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        B = 4
        
        results, loss = losses.compute_supervised_loss(sample_args, pred, labels, B)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert len(results) == 6

    def test_compute_supervised_loss_pred_future(self, sample_args):
        """Test supervised loss with future prediction."""
        sample_args.hierarchical_labels = False
        sample_args.pred_future = True
        sample_args.num_seq = 5
        
        B = 4
        pred = torch.randn(B * sample_args.num_seq, 10)
        labels = torch.randint(0, 10, (B, sample_args.num_seq))
        
        results, loss = losses.compute_supervised_loss(sample_args, pred, labels, B)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert len(results) == 6

    def test_compute_supervised_loss_temporal_labels(self, sample_args):
        """Test supervised loss with temporal labels."""
        sample_args.hierarchical_labels = False
        sample_args.pred_future = False
        sample_args.num_seq = 5
        
        B = 4
        pred = torch.randn(B * sample_args.num_seq, 10)
        labels = torch.randint(0, 10, (B, sample_args.num_seq))
        
        results, loss = losses.compute_supervised_loss(sample_args, pred, labels, B)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert len(results) == 6

    def test_compute_supervised_loss_ignore_index(self, sample_args):
        """Test supervised loss with ignore_index."""
        sample_args.hierarchical_labels = False
        sample_args.pred_future = False
        
        pred = torch.randn(4, 10)
        labels = torch.tensor([0, 1, 2, -1])  # -1 should be ignored
        B = 4
        
        results, loss = losses.compute_supervised_loss(sample_args, pred, labels, B)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert len(results) == 6

    def test_compute_supervised_loss_different_shapes(self, sample_args):
        """Test supervised loss with different input shapes."""
        sample_args.hierarchical_labels = False
        sample_args.pred_future = False
        sample_args.num_seq = 5
        
        # Test case where labels have fewer samples than predictions
        B = 4
        pred = torch.randn(B * sample_args.num_seq, 10)
        labels = torch.randint(0, 10, (B,))
        
        results, loss = losses.compute_supervised_loss(sample_args, pred, labels, B)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert len(results) == 6


class TestComputeSelfsupervisedLoss:
    """Test cases for the compute_selfsupervised_loss function."""

    def test_compute_selfsupervised_loss_basic(self, sample_args):
        """Test basic self-supervised loss computation."""
        pred = torch.randn(4, 256)
        feature_dist = torch.randn(4, 4)
        target = torch.randn(4, 256)
        sizes_pred = torch.tensor([4, 4, 4, 4])
        sizes_mask = torch.ones(4, 4)
        B = 4
        
        results, loss = losses.compute_selfsupervised_loss(
            sample_args, pred, feature_dist, target, sizes_pred, sizes_mask, B
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert len(results) == 6

    def test_compute_selfsupervised_loss_different_sizes(self, sample_args):
        """Test self-supervised loss with different sizes."""
        sizes = [2, 4, 8]
        
        for size in sizes:
            pred = torch.randn(size, 256)
            feature_dist = torch.randn(size, size)
            target = torch.randn(size, 256)
            sizes_pred = torch.tensor([size] * size)
            sizes_mask = torch.ones(size, size)
            B = size
            
            results, loss = losses.compute_selfsupervised_loss(
                sample_args, pred, feature_dist, target, sizes_pred, sizes_mask, B
            )
            
            assert isinstance(loss, torch.Tensor)
            assert loss.item() >= 0
            assert len(results) == 6


class TestComputeScores:
    """Test cases for the compute_scores function."""

    def test_compute_scores_basic(self, sample_args):
        """Test basic score computation."""
        pred = torch.randn(4, 256)
        feature_dist = torch.randn(4, 4)
        sizes = torch.tensor([4, 4, 4, 4])
        B = 4
        
        pos_score, neg_score = losses.compute_scores(sample_args, pred, feature_dist, sizes, B)
        
        assert isinstance(pos_score, torch.Tensor)
        assert isinstance(neg_score, torch.Tensor)
        assert pos_score.shape == neg_score.shape

    def test_compute_scores_different_distances(self, sample_args):
        """Test score computation with different distance functions."""
        distances = ['regular', 'hyperbolic']
        
        for distance in distances:
            sample_args.distance = distance
            sample_args.hyperbolic = (distance == 'hyperbolic')
            
            pred = torch.randn(4, 256)
            feature_dist = torch.randn(4, 4)
            sizes = torch.tensor([4, 4, 4, 4])
            B = 4
            
            pos_score, neg_score = losses.compute_scores(sample_args, pred, feature_dist, sizes, B)
            
            assert isinstance(pos_score, torch.Tensor)
            assert isinstance(neg_score, torch.Tensor)

    def test_compute_scores_different_batch_sizes(self, sample_args):
        """Test score computation with different batch sizes."""
        batch_sizes = [1, 2, 4, 8]
        
        for B in batch_sizes:
            pred = torch.randn(B, 256)
            feature_dist = torch.randn(B, B)
            sizes = torch.tensor([B] * B)
            
            pos_score, neg_score = losses.compute_scores(sample_args, pred, feature_dist, sizes, B)
            
            assert isinstance(pos_score, torch.Tensor)
            assert isinstance(neg_score, torch.Tensor)
            assert pos_score.shape[0] == B
            assert neg_score.shape[0] == B


class TestComputeMask:
    """Test cases for the compute_mask function."""

    def test_compute_mask_basic(self, sample_args):
        """Test basic mask computation."""
        sizes = torch.tensor([4, 4, 4, 4])
        B = 4
        
        mask = losses.compute_mask(sample_args, sizes, B)
        
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (B, B)
        assert mask.dtype == torch.bool

    def test_compute_mask_different_sizes(self, sample_args):
        """Test mask computation with different sizes."""
        batch_sizes = [1, 2, 4, 8]
        
        for B in batch_sizes:
            sizes = torch.tensor([B] * B)
            mask = losses.compute_mask(sample_args, sizes, B)
            
            assert isinstance(mask, torch.Tensor)
            assert mask.shape == (B, B)
            assert mask.dtype == torch.bool

    def test_compute_mask_asymmetric_sizes(self, sample_args):
        """Test mask computation with asymmetric sizes."""
        sizes = torch.tensor([2, 4, 6, 8])
        B = 4
        
        mask = losses.compute_mask(sample_args, sizes, B)
        
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (B, B)
        assert mask.dtype == torch.bool

    def test_compute_mask_early_action(self, sample_args):
        """Test mask computation with early action."""
        sample_args.early_action = True
        
        sizes = torch.tensor([4, 4, 4, 4])
        B = 4
        
        mask = losses.compute_mask(sample_args, sizes, B)
        
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (B, B)
        assert mask.dtype == torch.bool

    def test_compute_mask_early_action_self(self, sample_args):
        """Test mask computation with early action self."""
        sample_args.early_action_self = True
        
        sizes = torch.tensor([4, 4, 4, 4])
        B = 4
        
        mask = losses.compute_mask(sample_args, sizes, B)
        
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (B, B)
        assert mask.dtype == torch.bool


class TestBookkeeping:
    """Test cases for the bookkeeping function."""

    def test_bookkeeping_basic(self, sample_args):
        """Test basic bookkeeping functionality."""
        # Create mock AverageMeter objects
        avg_meters = {
            'losses': AverageMeter(),
            'accuracy': AverageMeter(),
            'hier_accuracy': AverageMeter(),
            'top1': AverageMeter(),
            'top3': AverageMeter(),
            'top5': AverageMeter()
        }
        
        # Mock results
        results = [0.5, 0.8, 0.7, 0.8, 0.9, 0.95]
        
        # Call bookkeeping
        losses.bookkeeping(sample_args, avg_meters, results)
        
        # Check that meters were updated
        assert avg_meters['losses'].avg == 0.5
        assert avg_meters['accuracy'].avg == 0.8
        assert avg_meters['hier_accuracy'].avg == 0.7
        assert avg_meters['top1'].avg == 0.8
        assert avg_meters['top3'].avg == 0.9
        assert avg_meters['top5'].avg == 0.95

    def test_bookkeeping_multiple_updates(self, sample_args):
        """Test bookkeeping with multiple updates."""
        avg_meters = {
            'losses': AverageMeter(),
            'accuracy': AverageMeter(),
            'hier_accuracy': AverageMeter(),
            'top1': AverageMeter(),
            'top3': AverageMeter(),
            'top5': AverageMeter()
        }
        
        # Multiple updates
        results1 = [0.5, 0.8, 0.7, 0.8, 0.9, 0.95]
        results2 = [0.3, 0.9, 0.8, 0.9, 0.95, 0.98]
        
        losses.bookkeeping(sample_args, avg_meters, results1)
        losses.bookkeeping(sample_args, avg_meters, results2)
        
        # Check that averages are computed correctly
        assert avg_meters['losses'].avg == 0.4  # (0.5 + 0.3) / 2
        assert avg_meters['accuracy'].avg == 0.85  # (0.8 + 0.9) / 2

    def test_bookkeeping_with_nan_values(self, sample_args):
        """Test bookkeeping with NaN values."""
        avg_meters = {
            'losses': AverageMeter(),
            'accuracy': AverageMeter(),
            'hier_accuracy': AverageMeter(),
            'top1': AverageMeter(),
            'top3': AverageMeter(),
            'top5': AverageMeter()
        }
        
        # Results with NaN (should be handled gracefully)
        results = [0.5, float('nan'), 0.7, 0.8, 0.9, 0.95]
        
        # Should not crash
        losses.bookkeeping(sample_args, avg_meters, results)
        
        # Check that non-NaN values are still processed
        assert avg_meters['losses'].avg == 0.5


class TestLossIntegration:
    """Integration tests for the loss functions."""

    def test_loss_gradients_supervised(self, sample_args):
        """Test that supervised loss produces gradients."""
        sample_args.use_labels = True
        
        # Create model parameters that require gradients
        pred = torch.randn(4, 10, requires_grad=True)
        labels = torch.randint(0, 10, (4,))
        feature_dist = torch.randn(4, 4)
        target = torch.randn(4, 256)
        sizes_pred = torch.tensor([4, 4, 4, 4])
        sizes_mask = torch.ones(4, 4)
        B = 4
        
        results = losses.compute_loss(
            sample_args, feature_dist, pred, labels, target, sizes_pred, sizes_mask, B
        )
        
        loss = results[0]
        loss.backward()
        
        # Check that gradients were computed
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_loss_gradients_selfsupervised(self, sample_args):
        """Test that self-supervised loss produces gradients."""
        sample_args.use_labels = False
        
        # Create model parameters that require gradients
        pred = torch.randn(4, 256, requires_grad=True)
        feature_dist = torch.randn(4, 4, requires_grad=True)
        target = torch.randn(4, 256)
        labels = torch.randint(0, 10, (4,))
        sizes_pred = torch.tensor([4, 4, 4, 4])
        sizes_mask = torch.ones(4, 4)
        B = 4
        
        results = losses.compute_loss(
            sample_args, feature_dist, pred, labels, target, sizes_pred, sizes_mask, B
        )
        
        loss = results[0]
        loss.backward()
        
        # Check that gradients were computed
        assert pred.grad is not None
        assert feature_dist.grad is not None
        assert not torch.isnan(pred.grad).any()
        assert not torch.isnan(feature_dist.grad).any()

    def test_loss_consistency(self, sample_args):
        """Test loss consistency across multiple runs."""
        sample_args.use_labels = True
        
        # Fixed inputs
        torch.manual_seed(42)
        pred = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        feature_dist = torch.randn(4, 4)
        target = torch.randn(4, 256)
        sizes_pred = torch.tensor([4, 4, 4, 4])
        sizes_mask = torch.ones(4, 4)
        B = 4
        
        # Run multiple times
        results1 = losses.compute_loss(
            sample_args, feature_dist, pred, labels, target, sizes_pred, sizes_mask, B
        )
        results2 = losses.compute_loss(
            sample_args, feature_dist, pred, labels, target, sizes_pred, sizes_mask, B
        )
        
        # Results should be identical
        for r1, r2 in zip(results1, results2):
            assert torch.allclose(r1, r2, atol=1e-6)

    def test_loss_numerical_stability(self, sample_args):
        """Test loss numerical stability with extreme values."""
        sample_args.use_labels = True
        
        # Test with very large values
        pred = torch.randn(4, 10) * 100
        labels = torch.randint(0, 10, (4,))
        feature_dist = torch.randn(4, 4)
        target = torch.randn(4, 256)
        sizes_pred = torch.tensor([4, 4, 4, 4])
        sizes_mask = torch.ones(4, 4)
        B = 4
        
        results = losses.compute_loss(
            sample_args, feature_dist, pred, labels, target, sizes_pred, sizes_mask, B
        )
        
        # Loss should still be finite
        assert torch.isfinite(results[0])
        
        # Test with very small values
        pred = torch.randn(4, 10) * 1e-6
        results = losses.compute_loss(
            sample_args, feature_dist, pred, labels, target, sizes_pred, sizes_mask, B
        )
        
        # Loss should still be finite
        assert torch.isfinite(results[0])
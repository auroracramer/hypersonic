import pytest
import torch
import numpy as np
import math
import sys
import os

# Add the parent directory to sys.path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.poincare_distance import poincare_distance, square_norm, pairwise_distances


class TestPoincareDistance:
    """Test cases for the poincare_distance function."""

    def test_poincare_distance_basic(self):
        """Test basic poincare distance calculation."""
        # Create simple test points
        pred = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        gt = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        
        # Calculate distance
        dist = poincare_distance(pred, gt)
        
        # Check shape
        assert dist.shape == (2, 2)
        
        # Distance from point to itself should be 0 (or very close to 0)
        assert torch.allclose(torch.diag(dist), torch.zeros(2), atol=1e-6)
        
        # Distance should be positive
        assert (dist >= 0).all()

    def test_poincare_distance_different_shapes(self):
        """Test poincare distance with different input shapes."""
        # Test different numbers of points
        shapes = [(1, 2), (3, 2), (5, 2), (10, 2)]
        
        for n_pred, n_gt in [(3, 5), (5, 3), (1, 10), (10, 1)]:
            pred = torch.randn(n_pred, 2) * 0.1  # Keep points small to stay in unit disk
            gt = torch.randn(n_gt, 2) * 0.1
            
            dist = poincare_distance(pred, gt)
            
            # Check output shape
            assert dist.shape == (n_pred, n_gt)
            
            # Check that all distances are positive
            assert (dist >= 0).all()

    def test_poincare_distance_different_dimensions(self):
        """Test poincare distance with different dimensions."""
        dimensions = [2, 3, 5, 10]
        
        for dim in dimensions:
            pred = torch.randn(3, dim) * 0.1
            gt = torch.randn(2, dim) * 0.1
            
            dist = poincare_distance(pred, gt)
            
            assert dist.shape == (3, 2)
            assert (dist >= 0).all()

    def test_poincare_distance_boundary_cases(self):
        """Test poincare distance with boundary cases."""
        # Test with points close to unit circle boundary
        pred = torch.tensor([[0.9, 0.0], [0.0, 0.9]])
        gt = torch.tensor([[0.9, 0.0], [0.0, 0.9]])
        
        dist = poincare_distance(pred, gt)
        
        # Should still work without NaN or inf
        assert torch.isfinite(dist).all()
        assert (dist >= 0).all()

    def test_poincare_distance_zero_points(self):
        """Test poincare distance with zero points."""
        pred = torch.zeros(2, 2)
        gt = torch.zeros(2, 2)
        
        dist = poincare_distance(pred, gt)
        
        # Distance between zero points should be 0
        assert torch.allclose(dist, torch.zeros(2, 2), atol=1e-6)

    def test_poincare_distance_symmetry(self):
        """Test poincare distance symmetry."""
        pred = torch.randn(3, 2) * 0.1
        gt = torch.randn(3, 2) * 0.1
        
        dist1 = poincare_distance(pred, gt)
        dist2 = poincare_distance(gt, pred)
        
        # Distance should be symmetric
        assert torch.allclose(dist1, dist2.T, atol=1e-6)

    def test_poincare_distance_triangle_inequality(self):
        """Test poincare distance triangle inequality."""
        # Create three points
        a = torch.tensor([[0.1, 0.1]])
        b = torch.tensor([[0.2, 0.2]])
        c = torch.tensor([[0.3, 0.3]])
        
        # Calculate distances
        dist_ab = poincare_distance(a, b)
        dist_bc = poincare_distance(b, c)
        dist_ac = poincare_distance(a, c)
        
        # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        assert (dist_ac <= dist_ab + dist_bc + 1e-6).all()  # Small tolerance for numerical errors

    def test_poincare_distance_numerical_stability(self):
        """Test poincare distance numerical stability."""
        # Test with very small values
        pred = torch.randn(3, 2) * 1e-6
        gt = torch.randn(3, 2) * 1e-6
        
        dist = poincare_distance(pred, gt)
        
        assert torch.isfinite(dist).all()
        assert (dist >= 0).all()
        
        # Test with values close to 1 (but not exactly 1)
        pred = torch.tensor([[0.999, 0.0], [0.0, 0.999]])
        gt = torch.tensor([[0.999, 0.0], [0.0, 0.999]])
        
        dist = poincare_distance(pred, gt)
        
        assert torch.isfinite(dist).all()
        assert (dist >= 0).all()

    def test_poincare_distance_batch_processing(self):
        """Test poincare distance with batch processing."""
        batch_sizes = [1, 5, 10, 100]
        
        for batch_size in batch_sizes:
            pred = torch.randn(batch_size, 2) * 0.1
            gt = torch.randn(batch_size, 2) * 0.1
            
            dist = poincare_distance(pred, gt)
            
            assert dist.shape == (batch_size, batch_size)
            assert torch.isfinite(dist).all()
            assert (dist >= 0).all()

    def test_poincare_distance_gradient_computation(self):
        """Test poincare distance gradient computation."""
        pred = torch.randn(2, 2, requires_grad=True) * 0.1
        gt = torch.randn(2, 2) * 0.1
        
        dist = poincare_distance(pred, gt)
        loss = dist.sum()
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients exist and are finite
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestSquareNorm:
    """Test cases for the square_norm function."""

    def test_square_norm_basic(self):
        """Test basic square norm calculation."""
        x = torch.tensor([[3.0, 4.0]])  # Norm should be 5, square norm should be 25
        
        norm_sq = square_norm(x)
        
        assert norm_sq.shape == (1,)
        assert torch.allclose(norm_sq, torch.tensor([25.0]))

    def test_square_norm_multiple_vectors(self):
        """Test square norm with multiple vectors."""
        x = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])
        
        norm_sq = square_norm(x)
        
        assert norm_sq.shape == (3,)
        expected = torch.tensor([25.0, 0.0, 2.0])
        assert torch.allclose(norm_sq, expected)

    def test_square_norm_different_dimensions(self):
        """Test square norm with different dimensions."""
        dimensions = [1, 2, 3, 5, 10]
        
        for dim in dimensions:
            x = torch.randn(3, dim)
            norm_sq = square_norm(x)
            
            assert norm_sq.shape == (3,)
            assert (norm_sq >= 0).all()  # Square norm should be non-negative

    def test_square_norm_zero_vector(self):
        """Test square norm with zero vector."""
        x = torch.zeros(2, 3)
        
        norm_sq = square_norm(x)
        
        assert norm_sq.shape == (2,)
        assert torch.allclose(norm_sq, torch.zeros(2))

    def test_square_norm_clamping(self):
        """Test square norm clamping."""
        # Create very small vector that might cause numerical issues
        x = torch.tensor([[1e-10, 1e-10]])
        
        norm_sq = square_norm(x)
        
        # Check that clamping works (minimum value should be 1e-5)
        assert norm_sq >= 1e-5
        assert norm_sq.shape == (1,)

    def test_square_norm_batch_processing(self):
        """Test square norm with batch processing."""
        batch_sizes = [1, 5, 10, 100]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 5)
            norm_sq = square_norm(x)
            
            assert norm_sq.shape == (batch_size,)
            assert (norm_sq >= 0).all()

    def test_square_norm_gradient_computation(self):
        """Test square norm gradient computation."""
        x = torch.randn(3, 2, requires_grad=True)
        
        norm_sq = square_norm(x)
        loss = norm_sq.sum()
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients exist and are finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_square_norm_consistency(self):
        """Test square norm consistency with torch.norm."""
        x = torch.randn(5, 3)
        
        # Calculate using our function
        norm_sq = square_norm(x)
        
        # Calculate using torch.norm
        torch_norm_sq = torch.norm(x, dim=-1, p=2) ** 2
        
        # Should be approximately equal (accounting for our clamping)
        assert torch.allclose(norm_sq, torch.clamp(torch_norm_sq, min=1e-5))


class TestPairwiseDistances:
    """Test cases for the pairwise_distances function."""

    def test_pairwise_distances_basic(self):
        """Test basic pairwise distances calculation."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        dist = pairwise_distances(x)
        
        assert dist.shape == (2, 2)
        assert torch.allclose(torch.diag(dist), torch.zeros(2))  # Distance to self should be 0

    def test_pairwise_distances_with_y(self):
        """Test pairwise distances with separate y."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        
        dist = pairwise_distances(x, y)
        
        assert dist.shape == (2, 3)
        assert (dist >= 0).all()

    def test_pairwise_distances_symmetry(self):
        """Test pairwise distances symmetry."""
        x = torch.randn(3, 2)
        
        dist = pairwise_distances(x)
        
        # Distance matrix should be symmetric
        assert torch.allclose(dist, dist.T, atol=1e-6)

    def test_pairwise_distances_different_shapes(self):
        """Test pairwise distances with different shapes."""
        shapes = [(1, 2), (3, 2), (5, 2), (10, 2)]
        
        for n_x, n_y in [(3, 5), (5, 3), (1, 10), (10, 1)]:
            x = torch.randn(n_x, 2)
            y = torch.randn(n_y, 2)
            
            dist = pairwise_distances(x, y)
            
            assert dist.shape == (n_x, n_y)
            assert (dist >= 0).all()

    def test_pairwise_distances_different_dimensions(self):
        """Test pairwise distances with different dimensions."""
        dimensions = [1, 2, 3, 5, 10]
        
        for dim in dimensions:
            x = torch.randn(3, dim)
            y = torch.randn(2, dim)
            
            dist = pairwise_distances(x, y)
            
            assert dist.shape == (3, 2)
            assert (dist >= 0).all()

    def test_pairwise_distances_zero_vectors(self):
        """Test pairwise distances with zero vectors."""
        x = torch.zeros(2, 3)
        y = torch.zeros(2, 3)
        
        dist = pairwise_distances(x, y)
        
        assert dist.shape == (2, 2)
        assert torch.allclose(dist, torch.zeros(2, 2))

    def test_pairwise_distances_single_point(self):
        """Test pairwise distances with single points."""
        x = torch.tensor([[1.0, 2.0]])
        y = torch.tensor([[3.0, 4.0]])
        
        dist = pairwise_distances(x, y)
        
        assert dist.shape == (1, 1)
        expected = torch.tensor([[8.0]])  # (1-3)^2 + (2-4)^2 = 4 + 4 = 8
        assert torch.allclose(dist, expected)

    def test_pairwise_distances_clamping(self):
        """Test pairwise distances clamping."""
        # Create points that might cause numerical issues
        x = torch.tensor([[1e-10, 1e-10], [1e-10, 1e-10]])
        
        dist = pairwise_distances(x)
        
        # Check that clamping works (minimum value should be 1e-7)
        assert (dist >= 1e-7).all()
        assert dist.shape == (2, 2)

    def test_pairwise_distances_batch_processing(self):
        """Test pairwise distances with batch processing."""
        batch_sizes = [1, 5, 10, 100]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3)
            
            dist = pairwise_distances(x)
            
            assert dist.shape == (batch_size, batch_size)
            assert (dist >= 0).all()

    def test_pairwise_distances_gradient_computation(self):
        """Test pairwise distances gradient computation."""
        x = torch.randn(3, 2, requires_grad=True)
        y = torch.randn(2, 2)
        
        dist = pairwise_distances(x, y)
        loss = dist.sum()
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients exist and are finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_pairwise_distances_consistency(self):
        """Test pairwise distances consistency with manual calculation."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        
        dist = pairwise_distances(x, y)
        
        # Manual calculation
        # x[0] to y[0]: (1-0)^2 + (2-0)^2 = 1 + 4 = 5
        # x[0] to y[1]: (1-1)^2 + (2-1)^2 = 0 + 1 = 1
        # x[1] to y[0]: (3-0)^2 + (4-0)^2 = 9 + 16 = 25
        # x[1] to y[1]: (3-1)^2 + (4-1)^2 = 4 + 9 = 13
        
        expected = torch.tensor([[5.0, 1.0], [25.0, 13.0]])
        assert torch.allclose(dist, expected)

    def test_pairwise_distances_edge_cases(self):
        """Test pairwise distances edge cases."""
        # Test with very large values
        x = torch.tensor([[1e6, 1e6]])
        y = torch.tensor([[0.0, 0.0]])
        
        dist = pairwise_distances(x, y)
        
        assert torch.isfinite(dist).all()
        assert (dist >= 0).all()
        
        # Test with very small values
        x = torch.tensor([[1e-6, 1e-6]])
        y = torch.tensor([[0.0, 0.0]])
        
        dist = pairwise_distances(x, y)
        
        assert torch.isfinite(dist).all()
        assert (dist >= 0).all()


class TestPoincareDistanceIntegration:
    """Integration tests for poincare distance functions."""

    def test_poincare_distance_components_integration(self):
        """Test integration of all poincare distance components."""
        pred = torch.randn(3, 2) * 0.1
        gt = torch.randn(4, 2) * 0.1
        
        # Test that all components work together
        dist = poincare_distance(pred, gt)
        
        # Verify intermediate calculations
        pred_norm = square_norm(pred)
        gt_norm = square_norm(gt)
        pairwise_dist = pairwise_distances(pred, gt)
        
        # All should be valid
        assert torch.isfinite(pred_norm).all()
        assert torch.isfinite(gt_norm).all()
        assert torch.isfinite(pairwise_dist).all()
        assert torch.isfinite(dist).all()
        
        # Final distance should be positive
        assert (dist >= 0).all()

    def test_poincare_distance_mathematical_properties(self):
        """Test mathematical properties of poincare distance."""
        # Test with known points
        origin = torch.zeros(1, 2)
        point1 = torch.tensor([[0.5, 0.0]])
        point2 = torch.tensor([[0.0, 0.5]])
        
        # Distance from origin to any point
        dist_origin_1 = poincare_distance(origin, point1)
        dist_origin_2 = poincare_distance(origin, point2)
        
        # Both should be positive and finite
        assert torch.isfinite(dist_origin_1).all()
        assert torch.isfinite(dist_origin_2).all()
        assert (dist_origin_1 > 0).all()
        assert (dist_origin_2 > 0).all()
        
        # Distance between symmetric points
        dist_1_2 = poincare_distance(point1, point2)
        
        assert torch.isfinite(dist_1_2).all()
        assert (dist_1_2 > 0).all()

    def test_poincare_distance_gradient_flow(self):
        """Test gradient flow through poincare distance."""
        # Create learnable parameters
        pred = torch.randn(2, 2, requires_grad=True) * 0.1
        gt = torch.randn(2, 2) * 0.1
        
        # Create optimizer
        optimizer = torch.optim.Adam([pred], lr=0.01)
        
        # Run a few optimization steps
        for _ in range(5):
            optimizer.zero_grad()
            
            dist = poincare_distance(pred, gt)
            loss = dist.sum()
            
            loss.backward()
            
            # Check gradients are finite
            assert torch.isfinite(pred.grad).all()
            
            optimizer.step()
        
        # Optimization should have run without errors
        assert True

    def test_poincare_distance_cuda_compatibility(self):
        """Test CUDA compatibility if available."""
        if torch.cuda.is_available():
            pred = torch.randn(3, 2, device='cuda') * 0.1
            gt = torch.randn(4, 2, device='cuda') * 0.1
            
            dist = poincare_distance(pred, gt)
            
            assert dist.device.type == 'cuda'
            assert torch.isfinite(dist).all()
            assert (dist >= 0).all()

    def test_poincare_distance_numerical_precision(self):
        """Test numerical precision across different dtypes."""
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            pred = torch.randn(2, 2, dtype=dtype) * 0.1
            gt = torch.randn(2, 2, dtype=dtype) * 0.1
            
            dist = poincare_distance(pred, gt)
            
            assert dist.dtype == dtype
            assert torch.isfinite(dist).all()
            assert (dist >= 0).all()

    def test_poincare_distance_performance_scaling(self):
        """Test performance scaling with different sizes."""
        sizes = [10, 50, 100, 500]
        
        for size in sizes:
            pred = torch.randn(size, 2) * 0.1
            gt = torch.randn(size, 2) * 0.1
            
            # Should complete without memory issues
            dist = poincare_distance(pred, gt)
            
            assert dist.shape == (size, size)
            assert torch.isfinite(dist).all()
            assert (dist >= 0).all()

    def test_poincare_distance_with_extreme_values(self):
        """Test poincare distance with extreme values."""
        # Test with points very close to unit circle
        pred = torch.tensor([[0.9999, 0.0], [0.0, 0.9999]])
        gt = torch.tensor([[0.9999, 0.0], [0.0, 0.9999]])
        
        dist = poincare_distance(pred, gt)
        
        # Should handle extreme values gracefully
        assert torch.isfinite(dist).all()
        assert (dist >= 0).all()
        
        # Test with very small values
        pred = torch.tensor([[1e-10, 1e-10]])
        gt = torch.tensor([[1e-10, 1e-10]])
        
        dist = poincare_distance(pred, gt)
        
        assert torch.isfinite(dist).all()
        assert (dist >= 0).all()
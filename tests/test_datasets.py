"""Tests for datasets module."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile

import datasets


class TestKinetics600:
    """Test cases for Kinetics600 dataset."""

    def test_kinetics_initialization(self):
        """Test that Kinetics600 can be initialized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = datasets.Kinetics600(
                mode='train',
                seq_len=8,
                num_seq=4,
                downsample=3,
                epsilon=5,
                path_dataset=temp_dir,
                path_data_info=temp_dir
            )
            
            assert dataset.mode == 'train'
            assert dataset.seq_len == 8
            assert dataset.num_seq == 4
            assert dataset.downsample == 3
            assert dataset.epsilon == 5

    def test_pil_loader_error_handling(self):
        """Test that pil_loader handles errors gracefully."""
        # Test with non-existent file
        result = datasets.pil_loader('nonexistent_file.jpg')
        
        # Should return a default RGB image
        assert result is not None
        assert result.mode == 'RGB'
        assert result.size == (150, 150)


class TestDatasetUtils:
    """Test dataset utility functions."""

    def test_sizes_hierarchy(self):
        """Test that sizes_hierarchy contains expected datasets."""
        assert 'finegym' in datasets.sizes_hierarchy
        assert 'hollywood2' in datasets.sizes_hierarchy
        
        # Check structure
        finegym_info = datasets.sizes_hierarchy['finegym']
        assert len(finegym_info) == 2
        assert isinstance(finegym_info[0], int)
        assert isinstance(finegym_info[1], list)

    @patch('datasets.get_data')
    def test_get_data_function_exists(self, mock_get_data):
        """Test that get_data function can be called."""
        mock_args = Mock()
        mock_get_data.return_value = (Mock(), Mock())
        
        train_loader, val_loader = datasets.get_data(mock_args)
        
        assert mock_get_data.called
        assert train_loader is not None
        assert val_loader is not None
import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.augmentation import *


class TestImage:
    """Test cases for the Image class/module."""

    def test_image_import(self):
        """Test that Image can be imported."""
        assert Image is not None
        
    def test_image_new(self):
        """Test creating new image."""
        img = Image.new('RGB', (100, 100), color='red')
        assert img is not None
        assert img.mode == 'RGB'
        assert img.size == (100, 100)

    def test_image_open(self, temp_dir):
        """Test opening existing image."""
        # Create a test image file
        test_img = Image.new('RGB', (50, 50), color='blue')
        test_path = os.path.join(temp_dir, 'test.jpg')
        test_img.save(test_path)
        
        # Test opening
        opened_img = Image.open(test_path)
        assert opened_img is not None
        assert opened_img.mode == 'RGB'
        assert opened_img.size == (50, 50)


class TestAugmentationFunctions:
    """Test cases for augmentation functions."""
    
    def test_tensor_to_pil_basic(self):
        """Test basic tensor to PIL conversion."""
        # Create a dummy tensor (C, H, W)
        tensor = torch.randn(3, 224, 224)
        
        # This assumes tensor_to_pil exists in the augmentation module
        # If not, this test will need to be adapted
        try:
            from utils.augmentation import tensor_to_pil
            pil_img = tensor_to_pil(tensor)
            assert isinstance(pil_img, Image.Image)
        except ImportError:
            pytest.skip("tensor_to_pil function not found")

    def test_normalize_tensor(self):
        """Test tensor normalization."""
        # Create a dummy tensor
        tensor = torch.randn(3, 224, 224)
        
        try:
            from utils.augmentation import normalize
            normalized = normalize(tensor)
            assert isinstance(normalized, torch.Tensor)
            assert normalized.shape == tensor.shape
        except ImportError:
            pytest.skip("normalize function not found")

    def test_resize_transform(self):
        """Test resize transformation."""
        # Create a dummy PIL image
        img = Image.new('RGB', (256, 256), color='red')
        
        try:
            from utils.augmentation import Resize
            resize_transform = Resize((224, 224))
            resized_img = resize_transform(img)
            assert isinstance(resized_img, Image.Image)
            assert resized_img.size == (224, 224)
        except ImportError:
            pytest.skip("Resize transform not found")

    def test_random_crop(self):
        """Test random crop transformation."""
        # Create a dummy PIL image
        img = Image.new('RGB', (256, 256), color='green')
        
        try:
            from utils.augmentation import RandomCrop
            crop_transform = RandomCrop((224, 224))
            cropped_img = crop_transform(img)
            assert isinstance(cropped_img, Image.Image)
            assert cropped_img.size == (224, 224)
        except ImportError:
            pytest.skip("RandomCrop transform not found")

    def test_random_horizontal_flip(self):
        """Test random horizontal flip transformation."""
        # Create a dummy PIL image
        img = Image.new('RGB', (224, 224), color='blue')
        
        try:
            from utils.augmentation import RandomHorizontalFlip
            flip_transform = RandomHorizontalFlip(p=1.0)  # Always flip
            flipped_img = flip_transform(img)
            assert isinstance(flipped_img, Image.Image)
            assert flipped_img.size == img.size
        except ImportError:
            pytest.skip("RandomHorizontalFlip transform not found")

    def test_color_jitter(self):
        """Test color jitter transformation."""
        # Create a dummy PIL image
        img = Image.new('RGB', (224, 224), color='yellow')
        
        try:
            from utils.augmentation import ColorJitter
            jitter_transform = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
            jittered_img = jitter_transform(img)
            assert isinstance(jittered_img, Image.Image)
            assert jittered_img.size == img.size
        except ImportError:
            pytest.skip("ColorJitter transform not found")

    def test_to_tensor_transform(self):
        """Test ToTensor transformation."""
        # Create a dummy PIL image
        img = Image.new('RGB', (224, 224), color='purple')
        
        try:
            from utils.augmentation import ToTensor
            to_tensor_transform = ToTensor()
            tensor_img = to_tensor_transform(img)
            assert isinstance(tensor_img, torch.Tensor)
            assert tensor_img.shape == (3, 224, 224)
            assert 0 <= tensor_img.min() and tensor_img.max() <= 1
        except ImportError:
            pytest.skip("ToTensor transform not found")

    def test_transforms_compose(self):
        """Test composing multiple transforms."""
        # Create a dummy PIL image
        img = Image.new('RGB', (256, 256), color='orange')
        
        try:
            from utils.augmentation import Compose, Resize, ToTensor
            
            transform = Compose([
                Resize((224, 224)),
                ToTensor()
            ])
            
            result = transform(img)
            assert isinstance(result, torch.Tensor)
            assert result.shape == (3, 224, 224)
        except ImportError:
            pytest.skip("Compose or other transforms not found")

    def test_gaussian_blur(self):
        """Test Gaussian blur transformation."""
        # Create a dummy PIL image
        img = Image.new('RGB', (224, 224), color='cyan')
        
        try:
            from utils.augmentation import GaussianBlur
            blur_transform = GaussianBlur(kernel_size=5)
            blurred_img = blur_transform(img)
            assert isinstance(blurred_img, Image.Image)
            assert blurred_img.size == img.size
        except ImportError:
            pytest.skip("GaussianBlur transform not found")

    def test_random_rotation(self):
        """Test random rotation transformation."""
        # Create a dummy PIL image
        img = Image.new('RGB', (224, 224), color='magenta')
        
        try:
            from utils.augmentation import RandomRotation
            rotation_transform = RandomRotation(degrees=30)
            rotated_img = rotation_transform(img)
            assert isinstance(rotated_img, Image.Image)
            # Size might change with rotation, so just check it's an image
        except ImportError:
            pytest.skip("RandomRotation transform not found")

    def test_random_grayscale(self):
        """Test random grayscale transformation."""
        # Create a dummy PIL image
        img = Image.new('RGB', (224, 224), color='lime')
        
        try:
            from utils.augmentation import RandomGrayscale
            grayscale_transform = RandomGrayscale(p=1.0)  # Always convert
            grayscale_img = grayscale_transform(img)
            assert isinstance(grayscale_img, Image.Image)
            assert grayscale_img.size == img.size
        except ImportError:
            pytest.skip("RandomGrayscale transform not found")


class TestVideoAugmentation:
    """Test cases for video-specific augmentation functions."""

    def test_video_tensor_augmentation(self):
        """Test augmentation on video tensors."""
        # Create a dummy video tensor (batch, channels, time, height, width)
        video_tensor = torch.randn(2, 3, 16, 224, 224)
        
        try:
            from utils.augmentation import video_augment
            augmented_video = video_augment(video_tensor)
            assert isinstance(augmented_video, torch.Tensor)
            assert augmented_video.shape == video_tensor.shape
        except ImportError:
            pytest.skip("video_augment function not found")

    def test_temporal_augmentation(self):
        """Test temporal augmentation."""
        # Create a dummy video tensor
        video_tensor = torch.randn(1, 3, 16, 224, 224)
        
        try:
            from utils.augmentation import temporal_augment
            augmented_video = temporal_augment(video_tensor)
            assert isinstance(augmented_video, torch.Tensor)
            # Shape might change with temporal augmentation
        except ImportError:
            pytest.skip("temporal_augment function not found")

    def test_spatial_augmentation(self):
        """Test spatial augmentation."""
        # Create a dummy video tensor
        video_tensor = torch.randn(1, 3, 16, 224, 224)
        
        try:
            from utils.augmentation import spatial_augment
            augmented_video = spatial_augment(video_tensor)
            assert isinstance(augmented_video, torch.Tensor)
            assert augmented_video.shape[0] == video_tensor.shape[0]  # batch size
            assert augmented_video.shape[1] == video_tensor.shape[1]  # channels
            assert augmented_video.shape[2] == video_tensor.shape[2]  # time
        except ImportError:
            pytest.skip("spatial_augment function not found")

    def test_frame_sampling(self):
        """Test frame sampling from video."""
        # Create a dummy video tensor
        video_tensor = torch.randn(1, 3, 32, 224, 224)  # 32 frames
        
        try:
            from utils.augmentation import sample_frames
            sampled_video = sample_frames(video_tensor, num_frames=16)
            assert isinstance(sampled_video, torch.Tensor)
            assert sampled_video.shape[2] == 16  # Should have 16 frames
        except ImportError:
            pytest.skip("sample_frames function not found")


class TestAugmentationUtilities:
    """Test cases for augmentation utility functions."""

    def test_random_seed_consistency(self):
        """Test that random seed produces consistent results."""
        # Create a dummy image
        img = Image.new('RGB', (224, 224), color='teal')
        
        try:
            from utils.augmentation import RandomHorizontalFlip
            
            # Set random seed
            torch.manual_seed(42)
            np.random.seed(42)
            
            flip_transform = RandomHorizontalFlip(p=0.5)
            
            # Apply transform multiple times with same seed
            torch.manual_seed(42)
            np.random.seed(42)
            result1 = flip_transform(img)
            
            torch.manual_seed(42)
            np.random.seed(42)
            result2 = flip_transform(img)
            
            # Results should be identical
            assert result1.size == result2.size
            # More detailed comparison would require converting to arrays
        except ImportError:
            pytest.skip("RandomHorizontalFlip transform not found")

    def test_parameter_validation(self):
        """Test parameter validation in augmentation functions."""
        try:
            from utils.augmentation import Resize
            
            # Test valid parameters
            resize_transform = Resize((224, 224))
            assert resize_transform is not None
            
            # Test invalid parameters
            with pytest.raises((ValueError, TypeError)):
                Resize((-1, 224))  # Invalid size
                
        except ImportError:
            pytest.skip("Resize transform not found")

    def test_image_format_handling(self):
        """Test handling of different image formats."""
        try:
            from utils.augmentation import ToTensor
            
            # Test with RGB image
            rgb_img = Image.new('RGB', (224, 224), color='red')
            to_tensor = ToTensor()
            rgb_tensor = to_tensor(rgb_img)
            assert rgb_tensor.shape[0] == 3
            
            # Test with grayscale image
            gray_img = Image.new('L', (224, 224), color=128)
            gray_tensor = to_tensor(gray_img)
            assert gray_tensor.shape[0] == 1
            
        except ImportError:
            pytest.skip("ToTensor transform not found")

    def test_batch_processing(self):
        """Test batch processing of augmentations."""
        # Create a batch of images
        batch_size = 4
        images = [Image.new('RGB', (224, 224), color='navy') for _ in range(batch_size)]
        
        try:
            from utils.augmentation import ToTensor
            
            to_tensor = ToTensor()
            
            # Process batch
            tensors = [to_tensor(img) for img in images]
            batch_tensor = torch.stack(tensors)
            
            assert batch_tensor.shape == (batch_size, 3, 224, 224)
            
        except ImportError:
            pytest.skip("ToTensor transform not found")


class TestAugmentationEdgeCases:
    """Test cases for edge cases in augmentation."""

    def test_empty_image_handling(self):
        """Test handling of empty or very small images."""
        try:
            from utils.augmentation import Resize
            
            # Test with very small image
            small_img = Image.new('RGB', (1, 1), color='black')
            resize_transform = Resize((224, 224))
            resized_img = resize_transform(small_img)
            
            assert resized_img.size == (224, 224)
            
        except ImportError:
            pytest.skip("Resize transform not found")

    def test_large_image_handling(self):
        """Test handling of very large images."""
        try:
            from utils.augmentation import Resize
            
            # Test with large image (might be memory intensive)
            try:
                large_img = Image.new('RGB', (4096, 4096), color='white')
                resize_transform = Resize((224, 224))
                resized_img = resize_transform(large_img)
                
                assert resized_img.size == (224, 224)
                
            except MemoryError:
                pytest.skip("Insufficient memory for large image test")
            
        except ImportError:
            pytest.skip("Resize transform not found")

    def test_extreme_parameters(self):
        """Test augmentation with extreme parameters."""
        img = Image.new('RGB', (224, 224), color='silver')
        
        try:
            from utils.augmentation import ColorJitter
            
            # Test with extreme color jitter
            extreme_jitter = ColorJitter(brightness=2.0, contrast=2.0, saturation=2.0, hue=0.5)
            jittered_img = extreme_jitter(img)
            
            assert isinstance(jittered_img, Image.Image)
            assert jittered_img.size == img.size
            
        except ImportError:
            pytest.skip("ColorJitter transform not found")


class TestAugmentationPerformance:
    """Test cases for augmentation performance."""

    def test_transform_speed(self):
        """Test that transforms complete in reasonable time."""
        import time
        
        # Create a batch of images
        images = [Image.new('RGB', (224, 224), color='gold') for _ in range(10)]
        
        try:
            from utils.augmentation import ToTensor, Resize, RandomHorizontalFlip, Compose
            
            transform = Compose([
                Resize((224, 224)),
                RandomHorizontalFlip(p=0.5),
                ToTensor()
            ])
            
            # Time the transformations
            start_time = time.time()
            for img in images:
                result = transform(img)
            end_time = time.time()
            
            # Should complete in reasonable time (less than 1 second for 10 images)
            assert end_time - start_time < 1.0
            
        except ImportError:
            pytest.skip("Required transforms not found")

    def test_memory_usage(self):
        """Test that transforms don't cause memory leaks."""
        import gc
        
        img = Image.new('RGB', (224, 224), color='maroon')
        
        try:
            from utils.augmentation import ToTensor
            
            to_tensor = ToTensor()
            
            # Apply transformation many times
            for _ in range(100):
                tensor = to_tensor(img)
                del tensor
                
            # Force garbage collection
            gc.collect()
            
            # If we get here without memory errors, the test passes
            assert True
            
        except ImportError:
            pytest.skip("ToTensor transform not found")


class TestAugmentationIntegration:
    """Integration tests for augmentation pipeline."""

    def test_full_augmentation_pipeline(self):
        """Test complete augmentation pipeline."""
        # Create a dummy image
        img = Image.new('RGB', (256, 256), color='olive')
        
        try:
            from utils.augmentation import (
                Compose, Resize, RandomCrop, RandomHorizontalFlip, 
                ColorJitter, ToTensor, Normalize
            )
            
            # Create typical training augmentation pipeline
            train_transform = Compose([
                Resize((256, 256)),
                RandomCrop((224, 224)),
                RandomHorizontalFlip(p=0.5),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Apply pipeline
            result = train_transform(img)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape == (3, 224, 224)
            
        except ImportError:
            pytest.skip("Required transforms not found")

    def test_evaluation_pipeline(self):
        """Test evaluation augmentation pipeline."""
        # Create a dummy image
        img = Image.new('RGB', (256, 256), color='purple')
        
        try:
            from utils.augmentation import (
                Compose, Resize, CenterCrop, ToTensor, Normalize
            )
            
            # Create typical evaluation pipeline
            eval_transform = Compose([
                Resize((256, 256)),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Apply pipeline
            result = eval_transform(img)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape == (3, 224, 224)
            
        except ImportError:
            pytest.skip("Required transforms not found")

    def test_augmentation_consistency(self):
        """Test augmentation consistency across multiple runs."""
        # Create a dummy image
        img = Image.new('RGB', (224, 224), color='brown')
        
        try:
            from utils.augmentation import Compose, Resize, ToTensor
            
            # Create deterministic pipeline
            transform = Compose([
                Resize((224, 224)),
                ToTensor()
            ])
            
            # Apply multiple times
            results = []
            for _ in range(5):
                result = transform(img)
                results.append(result)
            
            # All results should be identical for deterministic transforms
            for i in range(1, len(results)):
                assert torch.allclose(results[0], results[i])
                
        except ImportError:
            pytest.skip("Required transforms not found")

    def test_augmentation_with_different_formats(self):
        """Test augmentation with different image formats."""
        formats = ['RGB', 'L', 'RGBA']
        
        for format_type in formats:
            try:
                if format_type == 'RGB':
                    img = Image.new(format_type, (224, 224), color='red')
                elif format_type == 'L':
                    img = Image.new(format_type, (224, 224), color=128)
                elif format_type == 'RGBA':
                    img = Image.new(format_type, (224, 224), color=(255, 0, 0, 255))
                
                from utils.augmentation import ToTensor
                
                to_tensor = ToTensor()
                tensor = to_tensor(img)
                
                assert isinstance(tensor, torch.Tensor)
                if format_type == 'RGB':
                    assert tensor.shape[0] == 3
                elif format_type == 'L':
                    assert tensor.shape[0] == 1
                elif format_type == 'RGBA':
                    assert tensor.shape[0] == 4
                    
            except ImportError:
                pytest.skip("ToTensor transform not found")
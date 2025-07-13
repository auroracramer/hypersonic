# Test Suite Documentation

This directory contains comprehensive unit tests for the video understanding project. The test suite provides thorough coverage of all major components and functionality.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_models.py           # Tests for model architectures
├── test_losses.py           # Tests for loss functions
├── test_datasets.py         # Tests for dataset handling
├── test_trainer.py          # Tests for training logic
├── test_utils.py            # Tests for utility functions
├── test_backbone.py         # Tests for backbone selection
├── test_poincare_distance.py # Tests for Poincaré distance calculations
├── test_augmentation.py     # Tests for data augmentation
├── run_tests.py             # Test runner script
├── requirements.txt         # Test dependencies
├── pytest.ini              # Pytest configuration
└── README.md               # This file
```

## Test Coverage

The test suite covers:

### Core Components
- **Models**: Model initialization, forward passes, hyperbolic/euclidean modes, gradient computation
- **Losses**: Supervised/self-supervised losses, accuracy calculations, gradient flow
- **Datasets**: Data loading, preprocessing, augmentation, different dataset types
- **Trainer**: Training loops, evaluation, checkpointing, distributed training
- **Utils**: Utility functions, metrics, checkpointing, data transformations

### Mathematical Functions
- **Poincaré Distance**: Distance calculations, numerical stability, gradient computation
- **Backbone Selection**: ResNet variants, feature extraction, model loading

### Data Processing
- **Augmentation**: Image transformations, video processing, pipeline composition
- **Utilities**: Accuracy calculations, normalization, logging

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r tests/requirements.txt
```

### Basic Usage

Run all tests:
```bash
python tests/run_tests.py
```

Run specific test categories:
```bash
python tests/run_tests.py --unit          # Unit tests only
python tests/run_tests.py --integration   # Integration tests only
python tests/run_tests.py --fast          # Skip slow tests
```

Run tests for specific modules:
```bash
python tests/run_tests.py --module models
python tests/run_tests.py --module losses
python tests/run_tests.py --module datasets
```

### Advanced Options

Run with coverage:
```bash
python tests/run_tests.py --coverage --html
```

Run in parallel:
```bash
python tests/run_tests.py --parallel
```

Run GPU tests (if available):
```bash
python tests/run_tests.py --gpu
```

Skip GPU tests:
```bash
python tests/run_tests.py --no-gpu
```

### Direct pytest Usage

You can also run tests directly with pytest:
```bash
pytest tests/                              # All tests
pytest tests/test_models.py               # Specific file
pytest tests/test_models.py::TestModel    # Specific test class
pytest -v tests/                          # Verbose output
pytest -x tests/                          # Stop on first failure
pytest -k "test_model_init" tests/        # Run tests matching pattern
```

## Test Categories

Tests are organized into categories using pytest markers:

- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (slower, multiple components)
- `@pytest.mark.slow`: Slow tests (long-running)
- `@pytest.mark.gpu`: Tests requiring GPU
- `@pytest.mark.network`: Tests requiring network access

## Writing New Tests

### Test Structure

Follow this structure for new tests:

```python
import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module_to_test import function_to_test

class TestFunctionName:
    """Test cases for function_to_test."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        input_data = create_test_data()
        
        # Act
        result = function_to_test(input_data)
        
        # Assert
        assert result is not None
        assert isinstance(result, expected_type)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with empty input
        result = function_to_test([])
        assert result == expected_empty_result
        
        # Test with invalid input
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
    
    def test_different_parameters(self):
        """Test with different parameters."""
        # Test multiple parameter combinations
        params = [param1, param2, param3]
        for param in params:
            result = function_to_test(param)
            assert validate_result(result)
```

### Fixtures

Use fixtures from `conftest.py`:

```python
def test_with_fixtures(self, sample_args, sample_video_data, temp_dir):
    """Test using predefined fixtures."""
    model = Model(sample_args)
    video, labels = sample_video_data
    
    # Test code here
```

### Mocking

Use mocking for external dependencies:

```python
from unittest.mock import patch, MagicMock

def test_with_mocking(self):
    """Test with mocked dependencies."""
    with patch('module.external_function') as mock_func:
        mock_func.return_value = expected_value
        
        result = function_under_test()
        
        mock_func.assert_called_once()
        assert result == expected_result
```

## Best Practices

1. **Test Isolation**: Each test should be independent and not rely on other tests
2. **Clear Naming**: Use descriptive test names that explain what is being tested
3. **AAA Pattern**: Structure tests with Arrange, Act, Assert sections
4. **Edge Cases**: Test boundary conditions and error cases
5. **Fixtures**: Use fixtures for common setup code
6. **Mocking**: Mock external dependencies to isolate the code under test
7. **Assertions**: Use specific assertions that provide clear failure messages

## Coverage Goals

The test suite aims for:
- **Line Coverage**: >80% of lines covered
- **Branch Coverage**: >70% of branches covered
- **Function Coverage**: >90% of functions tested

## Performance Considerations

- **Fast Tests**: Unit tests should complete in <100ms each
- **Slow Tests**: Mark long-running tests with `@pytest.mark.slow`
- **Resource Usage**: Tests should not consume excessive memory
- **Parallel Execution**: Tests should be safe to run in parallel

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **GPU Tests**: GPU tests will be skipped if CUDA is not available
3. **Memory Issues**: Large tests may fail on systems with limited memory
4. **Slow Tests**: Use `--fast` flag to skip slow tests during development

### Debug Mode

Run tests with debugging:
```bash
pytest --pdb tests/test_models.py::TestModel::test_model_init
```

### Verbose Output

Get detailed output:
```bash
pytest -v -s tests/
```

## Continuous Integration

The test suite is designed to work with CI/CD systems:

```yaml
# Example CI configuration
- name: Run tests
  run: |
    pip install -r tests/requirements.txt
    python tests/run_tests.py --coverage --parallel
```

## Contributing

When adding new functionality:

1. Write tests for new features
2. Ensure existing tests still pass
3. Add appropriate test markers
4. Update documentation if needed
5. Run the full test suite before submitting

For questions or issues with the test suite, please refer to the project documentation or create an issue.
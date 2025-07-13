# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-XX

### Added
- **Modern Project Structure**: Added `pyproject.toml` for modern Python packaging
- **Type Hints**: Added type annotations to main functions in `main.py`
- **Development Tools**: 
  - Pre-commit hooks configuration (`.pre-commit-config.yaml`)
  - GitHub Actions CI/CD workflow (`.github/workflows/ci.yml`)
  - Comprehensive `.gitignore` for Python/ML projects
- **Testing Framework**: 
  - Basic test structure in `tests/` directory
  - Unit tests for models, datasets, and utilities
  - pytest configuration with coverage reporting
- **Code Quality Tools**:
  - Black code formatter configuration
  - isort import sorting
  - flake8 linting
  - mypy type checking
- **Documentation**: 
  - Enhanced README.md with modern structure, badges, and comprehensive instructions
  - Package docstring in `__init__.py`
  - Basic usage example in `examples/basic_usage.py`
- **Development Dependencies**: Added optional development dependencies for testing and code quality

### Changed
- **Dependencies**: Updated all dependencies to modern versions
  - PyTorch: 1.7.0.dev → 2.0.0+
  - torchvision: 0.8.0.dev → 0.15.0+
  - numpy: 1.18.5 → 1.21.0+
  - All other dependencies updated to secure, modern versions
- **Requirements**: Switched from pinned versions to minimum version requirements
- **Documentation**: Completely restructured README.md with:
  - Modern badges and status indicators
  - Clear installation instructions
  - Comprehensive usage examples
  - Better organized sections
  - Development workflow documentation

### Improved
- **Code Quality**: Added consistent code formatting and linting
- **Developer Experience**: Added pre-commit hooks for automatic code quality checks
- **CI/CD**: Automated testing across Python 3.8-3.11
- **Package Structure**: Made the project properly installable as a Python package
- **Error Handling**: Better error handling in dataset loaders
- **Documentation**: More comprehensive and user-friendly documentation

### Technical Details
- **Python Support**: Requires Python 3.8+
- **Testing**: Comprehensive test suite with pytest
- **Packaging**: Modern packaging with setuptools and pyproject.toml
- **Code Style**: Enforced with black, isort, and flake8
- **Type Safety**: Optional type checking with mypy
- **CI/CD**: GitHub Actions workflow for automated testing

### Migration Guide
For existing users:

1. **Update Python**: Ensure you're using Python 3.8 or higher
2. **Install Updated Dependencies**: 
   ```bash
   pip install -e .[dev]  # For development
   pip install -e .       # For basic usage
   ```
3. **Update Scripts**: The main API remains the same, but some internal imports may need updates
4. **Development Setup**: If contributing, install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Compatibility
- **Breaking Changes**: Minimum Python version increased from 3.7 to 3.8
- **API Changes**: None - the main API remains backward compatible
- **Dependencies**: Updated dependencies may require different CUDA versions

### Credits
- Original implementation by Dídac Suris, Ruoshi Liu, and Carl Vondrick
- Modernization improvements added in 2024
- Built upon DPC and geoopt libraries

---

## [0.1.0] - 2021-01-XX (Original Release)

### Added
- Initial implementation of "Learning the Predictability of the Future"
- Support for Kinetics600, FineGym, MovieNet, and Hollywood2 datasets
- Hyperbolic geometry integration using geoopt
- Self-supervised learning framework
- Hierarchical action recognition
- ResNet-based feature extraction
- ConvGRU temporal modeling
- Training and evaluation scripts
# Learning the Predictability of the Future

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/HyperFuture/hyperfuture/actions/workflows/ci.yml/badge.svg)](https://github.com/HyperFuture/hyperfuture/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Code from the paper [Learning the Predictability of the Future](https://arxiv.org/abs/2101.01600).

🌐 **Website**: [hyperfuture.cs.columbia.edu](https://hyperfuture.cs.columbia.edu)  
📄 **Paper**: [arXiv:2101.01600](https://arxiv.org/abs/2101.01600)

## Overview

This project implements a framework for learning the predictability of the future using hyperbolic geometry. The model predicts how predictable future video frames will be, enabling better understanding of temporal dynamics in video sequences.

### Key Features

- **Hyperbolic Geometry**: Leverages hyperbolic space for better representation of hierarchical temporal structures
- **Multi-Dataset Support**: Works with Kinetics600, FineGym, MovieNet, and Hollywood2 datasets
- **Self-Supervised Learning**: Learns predictability without explicit supervision
- **Hierarchical Predictions**: Supports both action-level and sub-action-level predictions

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/HyperFuture/hyperfuture.git
cd hyperfuture

# Install in development mode
pip install -e .[dev]

# Or install just the package
pip install -e .
```

### Development Setup

For development, it's recommended to use the full development environment:

```bash
# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
flake8 .
black .
isort .
```

## Usage

### Basic Training

```bash
python main.py --dataset kinetics --batch_size 8 --epochs 100 --lr 1e-3
```

### Hyperbolic Training

```bash
python main.py --dataset kinetics --hyperbolic --batch_size 8 --epochs 100 --lr 1e-3
```

### Testing

```bash
python main.py --test --resume path/to/checkpoint.pth.tar
```

### Example Scripts

Under `scripts/` there are example bash files to run:
- Self-supervised training and finetuning
- Supervised training and testing

You will need to modify the paths to the datasets and dataset info folder.

### Command Line Arguments

Run `python main.py --help` for complete information on arguments.

Key arguments:
- `--dataset`: Choose from kinetics, finegym, movienet, hollywood2
- `--hyperbolic`: Enable hyperbolic geometry mode
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--path_dataset`: Path to dataset directory
- `--path_data_info`: Path to dataset info directory

## Datasets

We train our framework on four different datasets:

| Dataset | Description | Link |
|---------|-------------|------|
| [Kinetics600](https://deepmind.com/research/open-source/kinetics) | Large-scale video dataset | [Download](https://deepmind.com/research/open-source/kinetics) |
| [FineGym](https://sdolivia.github.io/FineGym/) | Fine-grained gym activity dataset | [Download](https://sdolivia.github.io/FineGym/) |
| [MovieNet](http://movienet.site) | Large-scale movie dataset | [Download](http://movienet.site) |
| [Hollywood2](https://www.di.ens.fr/~laptev/actions/hollywood2/) | Movie action dataset | [Download](https://www.di.ens.fr/~laptev/actions/hollywood2/) |

### Dataset Info

Additional dataset information (train/test splits, class hierarchies) can be found in:
[dataset_info.tar.gz](https://hyperfuture.cs.columbia.edu/dataset_info.tar.gz)

Extract with:
```bash
tar -xzvf dataset_info.tar.gz
```

Set the path using `--path_data_info`.

## Pretrained Models

Pretrained models from our paper are available at:
[checkpoints.tar.gz](https://hyperfuture.cs.columbia.edu/checkpoints.tar.gz)

Each folder contains a `.pth` file with the checkpoint.

To use pretrained models:
```bash
# Resume training
python main.py --resume path/to/checkpoint.pth

# Use as pretrained backbone
python main.py --pretrain path/to/checkpoint.pth
```

## Architecture

The model consists of:
- **Backbone**: ResNet-based feature extractor
- **Temporal Modeling**: ConvGRU for sequence processing
- **Hyperbolic Layers**: Optional hyperbolic neural networks
- **Prediction Head**: Future frame predictability estimation

## Development

### Code Quality

This project uses modern Python development practices:
- **Type Hints**: Full type annotations for better code clarity
- **Code Formatting**: Black for consistent code style
- **Import Sorting**: isort for organized imports
- **Linting**: flake8 for code quality
- **Testing**: pytest for comprehensive testing
- **CI/CD**: GitHub Actions for automated testing

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

### Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{suris2021hyperfuture,
    title={Learning the Predictability of the Future},
    author={Sur\'is, D\'idac and Liu, Ruoshi and Vondrick, Carl},
    journal={arXiv preprint arXiv:2101.01600},
    year={2021}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This code builds upon:
- [DPC](https://github.com/TengdaHan/DPC) for the base framework
- [geoopt](https://github.com/geoopt/geoopt) for hyperbolic operations

## Support

If you encounter any issues or have questions, please:
1. Check the [Issues](https://github.com/HyperFuture/hyperfuture/issues) page
2. Create a new issue with detailed information
3. Contact the authors via email

---

**Columbia University Computer Vision Lab** | **2021-2024**


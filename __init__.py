"""HyperFuture: Learning the Predictability of the Future using Hyperbolic Geometry.

This package implements a framework for learning the predictability of the future
using hyperbolic geometry for video understanding.
"""

__version__ = "1.0.0"
__author__ = "Dídac Suris, Ruoshi Liu, Carl Vondrick"
__email__ = "dsuris@cs.columbia.edu"

# Import main components
from . import models
from . import datasets
from . import losses
from . import trainer
from . import utils

__all__ = [
    "models",
    "datasets", 
    "losses",
    "trainer",
    "utils",
]
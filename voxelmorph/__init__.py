# ---- voxelmorph ----
# unsupervised learning for image registration

from . import generators
from . import py
from .py.utils import default_unet_features

try:
    import torch
except ImportError:
    raise ImportError('Please install pytorch to use this voxelmorph backend')

from . import torch
from .torch import layers
from .torch import networks
from .torch import losses

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



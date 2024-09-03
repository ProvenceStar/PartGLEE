from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .config import add_partglee_config
from .PartGLEE import PartGLEE
from .data import build_detection_train_loader, build_detection_test_loader
from .backbone.swin import D2SwinTransformer
# from .backbone.internimage import D2InternImage
from .backbone.eva02 import D2_EVA02
from .backbone.eva01 import D2_EVA01


from .motion3d2d import MotionEstimation3D2D
from ..utils import get_config


def MotionEstimation(**kwargs):
    config = get_config()
    if config["MotionEstimationMethod"].lower() == "3d2d":
        return MotionEstimation3D2D(**kwargs)
    else:
        raise NotImplementedError

from .flannmatcher import FlannMatcher
from ..utils import get_config

config = get_config()


def FeatureMatcher():
    if config["LocalFeatureMatcher"].lower() == "flann":
        return FlannMatcher()
    else:
        raise NotImplementedError()
import cv2
import numpy as np

from VSLAM.FeatureTrackers.Trackers import KLTTracker

from ..utils import get_config

config = get_config()


def FeatureTracker():
    if config["LocalFeatureTracker"] == "KLT":
        tracker = KLTTracker()
        tracker.type = "KLT"
        return tracker
    elif config["LocalFeatureTracker"] == "Matcher":
        tracker = MatcherTracker()
        tracker.type = "Matcher"
        return tracker
    else:
        raise NotImplementedError

from Datasets import Kitti 
from VSLAM import VisualSLAM
import matplotlib.pyplot as plt


ds = Kitti()

slam = VisualSLAM(ds.load_parameters())


for idx in range(len(ds)):
    inputs = ds.load_frame(idx)
    slam(inputs)
    #traj = slam.trajectory()
    #gt = ds.ground_truth()[:idx]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Datasets import Kitti
from VSLAM import VisualSLAM

ds = Kitti()
gt_all = ds.ground_truth()
slam = VisualSLAM(ds.load_parameters())

fig, ax = plt.subplots()
ax.set_title("VSLAM Tracking")

(line_traj,) = ax.plot([], [], "r-", label="Tracked")
(line_gt,) = ax.plot([], [], "g-", label="Ground Truth")

ax.set_xlim(np.min(gt_all[:, 2]) - 10, np.max(gt_all[:, 2]) + 10)
ax.set_ylim(np.min(gt_all[:, 0]) - 10, np.max(gt_all[:, 0]) + 10)

ax.set_xlabel("Z Position")
ax.set_ylabel("X Position")
ax.legend()


def update(frame_idx):
    inputs = ds.load_frame(frame_idx)
    slam(inputs)
    traj = slam.trajectory()
    gt = ds.ground_truth()[: frame_idx + 1]
    line_traj.set_data(traj[:, 2], traj[:, 0])
    line_gt.set_data(gt[:, 2], gt[:, 0])
    return line_traj, line_gt

for frame_idx in range(len(ds)):
    inputs = ds.load_frame(frame_idx)
    slam(inputs)

#ani = FuncAnimation(fig, update, frames=len(ds), blit=True, interval=100)
#plt.show()

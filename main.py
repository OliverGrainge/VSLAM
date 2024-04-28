import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Datasets import Kitti
from VSLAM import VisualSLAM

# Initialize dataset and SLAM system
ds = Kitti()
slam = VisualSLAM(ds.load_parameters())

# Create a figure and axis object
fig, ax = plt.subplots()

# Initialize lines for trajectory and ground truth
line_traj, = ax.plot([], [], 'r-', label="Tracked")
line_gt, = ax.plot([], [], 'g-', label="Ground Truth")

# Setting up plot limits, labels, and legend
ax.set_xlim(-10, 10)  # Set appropriate limits for your dataset
ax.set_ylim(-10, 10)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.legend()

def update(frame_idx):
    # Load frame data
    inputs = ds.load_frame(frame_idx)
    slam(inputs)
    
    # Get trajectory and ground truth up to the current frame
    traj = slam.trajectory()
    gt = ds.ground_truth()[:frame_idx+1]

    # Update the data of both lines
    line_traj.set_data(traj[:, 0], traj[:, 1])
    line_gt.set_data(gt[:, 0], gt[:, 1])

    # Optional: Update plot title to show frame number
    ax.set_title(f"Frame {frame_idx}")

    # Return the updated line objects
    return line_traj, line_gt

# Create animation
ani = FuncAnimation(fig, update, frames=len(ds), blit=True, interval=100)

# To show the animation in a Jupyter notebook, use the following:
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# To save the animation as a video file
# ani.save('slam_animation.mp4', writer='ffmpeg', fps=10)

# Display the plot
plt.show()
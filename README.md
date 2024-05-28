# VSLAM (Visual Simultaneous Localization and Mapping)

[![Watch the video](https://path_to_thumbnail_image.jpg)](https://github.com/OliverGrainge/VSLAM/assets/140703829/cced9a2f-f5e9-4844-886f-2b46847a6835)

The VSLAM repository is a simplified implementation of Stereo Visual SLAM, written entirely in Python. It uses numpy, opencv, and scipy for feature detection, tracking, matching, motion estimation, and optimization. The algorithm is compatible with the KITTI dataset and includes a comprehensive testing suite to ensure robustness.

Additionally, we provide demo scripts to illustrate key processes in the visual SLAM pipeline, such as matching, tracking, and motion estimation. Follow the steps below to set up and use the repository.

## Installation

1. **Clone the repository and navigate to the root directory:**

```bash
git clone https://github.com/OliverGrainge/VSLAM.git
cd VSLAM/
```

2. **Set up a new conda environment:**

```bash
conda create --name vslam_env python=3.10
conda activate vslam_env
```

3. **Install the required packages:**

```bash
python -m pip install -r requirements.txt
```

## Testing

After installing the packages, you can test the repository with pytest to ensure everything is working correctly:

```bash
pytest
```

## Demos

To understand how the individual components of VSLAM work, you can run the following scripts:

- **Point matching from left to right images:**

```bash
python examples/track_left_right_features.py
```

- **Consecutive point tracking:**

```bash
python examples/track_consecutive_features.py
```

- **Motion estimation:**

```bash
python examples/motion_estimation.py
```

## Running the Full Pipeline

To run the complete visual SLAM pipeline, execute:

```bash
python main.py
```
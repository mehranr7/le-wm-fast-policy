# le-wm-fast-policy

A fast and lightweight Behavioral Cloning (BC) framework designed to train **State-Based** and **Visual (ResNet)** offline policies for the DeepMind Control Reacher environment. This repository serves as an efficient, streamlined alternative to heavy generative World Models like LeWorldModel.

## Features
- **Dual Policy Architecture**: Train policies purely from proprioceptive physics state vectors or purely from raw visual camera pixels.
- **Offline Dataset Handling**: Includes tools to effortlessly parse and extract frames, rewards, and multi-dimensional state vectors directly from massive 98GB offline Reacher `.h5` datasets.
- **Dataset Visualization**: Automatically renders visual outputs (MP4) overlaid with target coordinates to diagnose and understand the underlying offline datasets.

## Project Structure
- `extract_episodes.py`: A utility script that scans the massive HDF5 dataset, parses terminal conditions (detecting blank `NaN` actions), and exports successful and failed episode data into JSON files and pristine MP4 videos.
- `data-instances/`: A collection of extracted test episodes, containing frame-by-frame json data and color-perfect `.mp4` video clips of the robotic arm.

### Sample Episodes
The dataset contains both successful and failed demonstrations of the robotic arm attempting to reach the red target.

**Failed Episodes (Reward = 0.0)**
- **Episode 0**: Failed to reach target. [View JSON Data](data-instances/ep000/data/)
  <br>
  <video src="data-instances/ep000/video.mp4" controls width="300"></video>

- **Episode 1**: Failed to reach target. [View JSON Data](data-instances/ep001/data/)
  <br>
  <video src="data-instances/ep001/video.mp4" controls width="300"></video>

**Successful Episodes (Reward > 0.0)**
- **Episode 7**: Successfully reached target. [View JSON Data](data-instances/ep007/data/)
  <br>
  <video src="data-instances/ep007/video.mp4" controls width="300"></video>

- **Episode 8**: Successfully reached target. [View JSON Data](data-instances/ep008/data/)
  <br>
  <video src="data-instances/ep008/video.mp4" controls width="300"></video>

## Upcoming Implementations
The following PyTorch modules are currently being implemented:
1. **`bc_dataset.py`**: A highly efficient HDF5 PyTorch DataLoader that parses 2-million frame datasets into batched `(3, 224, 224)` images or `(6,)` state vectors, whilst auto-filtering invalid terminal frames.
2. **`bc_model.py`**: Contains the architectures for `StateBCPolicy` (a dense MLP) and `VisualBCPolicy` (a ResNet-18 vision encoder fused with target coordinates).
3. **`train_bc.py`**: The training loop utilizing Mean Squared Error (MSE) backpropagation for exact behavioral cloning.

import h5py
import hdf5plugin
import os
import json
import numpy as np
from PIL import Image, ImageDraw
import math
import shutil

h5_path = '/data/5pourghe/le-wm-v2/le-wm-storage/reacher.h5'
base_out_dir = '/data/5pourghe/le-wm-v2/le-wm-storage/data-instances'

# Clean up existing data-instances directory to start fresh
if os.path.exists(base_out_dir):
    shutil.rmtree(base_out_dir)
os.makedirs(base_out_dir, exist_ok=True)

def draw_target(img, target_x, target_y):
    draw = ImageDraw.Draw(img)
    scale = 360.5
    px = int(112 + target_x * scale)
    py = int(112 - target_y * scale)
    r = 4
    draw.ellipse((px - r, py - r, px + r, py + r), fill=(255, 0, 0), outline=(255, 255, 255))
    return img

def safe_val(v):
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v

def safe_list(arr):
    if hasattr(arr, 'tolist'):
        arr = arr.tolist()
    if isinstance(arr, list):
        return [safe_val(x) for x in arr]
    return safe_val(arr)

print("Opening dataset...")
f = h5py.File(h5_path, 'r')

num_episodes = len(f['ep_len'])
ep_lengths = f['ep_len'][:]
ep_offsets = f['ep_offset'][:]

# Find 2 failed and 2 successful episodes
failed_episodes = []
successful_episodes = []

print("Scanning for failed and successful episodes...")
for ep_idx in range(num_episodes):
    if len(failed_episodes) == 2 and len(successful_episodes) == 2:
        break
        
    start_idx = int(ep_offsets[ep_idx])
    length = int(ep_lengths[ep_idx])
    
    # An episode is successful if it has any reward > 0
    rewards = f['reward'][start_idx : start_idx + length]
    is_success = np.any(rewards > 0)
    
    if is_success and len(successful_episodes) < 2:
        successful_episodes.append(ep_idx)
    elif not is_success and len(failed_episodes) < 2:
        failed_episodes.append(ep_idx)

print(f"Selected Failed Episodes: {failed_episodes}")
print(f"Selected Successful Episodes: {successful_episodes}")

target_episodes = failed_episodes + successful_episodes

keys_per_frame = [
    'action', 'ep_idx', 'finger_pos', 'id', 'observation', 
    'qpos', 'qvel', 'render_time', 'reward', 'score', 
    'step_idx', 'success', 'target_pos', 'terminated', 'truncated'
]

for ep_idx in target_episodes:
    # 1-indexed for naming, e.g. ep001
    ep_name = f"ep{(ep_idx + 1):03d}"
    out_dir = os.path.join(base_out_dir, ep_name)
    data_dir = os.path.join(out_dir, "data")
    img_dir = os.path.join(out_dir, "images")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    start_idx = int(ep_offsets[ep_idx])
    length = int(ep_lengths[ep_idx])
    status = "SUCCESS" if ep_idx in successful_episodes else "FAILED"
    
    print(f"Extracting {ep_name} [{status}] (Length: {length} frames)...")
    
    # Fetch all frame data for the episode at once for performance
    ep_data = {}
    for k in keys_per_frame:
        ep_data[k] = f[k][start_idx : start_idx + length]
        
    pixels_ep = f['pixels'][start_idx : start_idx + length]
    images_for_video = []
    
    for i in range(length):
        # Image Processing
        img_arr = pixels_ep[i]
        if img_arr.shape[0] == 3:
            img_arr = np.transpose(img_arr, (1, 2, 0))
        img = Image.fromarray(img_arr)
        
        t_x, t_y = ep_data['target_pos'][i]
        img = draw_target(img, t_x, t_y)
        images_for_video.append(img)
        
        # Save image (1-indexed formatting: f001.png)
        frame_name = f"f{(i + 1):03d}"
        img.save(os.path.join(img_dir, f"{frame_name}.png"))
        
        # Data JSON Processing
        frame_dict = {}
        for k in keys_per_frame:
            val = ep_data[k][i]
            if isinstance(val, (np.ndarray, list)):
                frame_dict[k] = safe_list(val)
            else:
                frame_dict[k] = safe_val(val)
                
        # Save JSON
        with open(os.path.join(data_dir, f"{frame_name}.json"), "w") as jf:
            json.dump(frame_dict, jf, indent=4)
            
    # Save Video
    gif_path = os.path.join(out_dir, "video.gif")
    images_for_video[0].save(
        gif_path, save_all=True, append_images=images_for_video[1:], loop=0, duration=20
    )

print("Extraction completely finished!")

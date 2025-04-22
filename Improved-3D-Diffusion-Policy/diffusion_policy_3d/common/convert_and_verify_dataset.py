#!/usr/bin/env python3

"""
Script to convert HDF5 dataset to Zarr format and verify the conversion
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import zarr
import h5py
from tqdm import tqdm
import logging

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.convert_hdf5_to_zarr import convert_hdf5_to_zarr

def visualize_hdf5_episode(hdf5_path, episode_idx=0):
    """
    Visualize a sample episode from the HDF5 file
    """
    with h5py.File(hdf5_path, "r", swmr=True, libver="latest") as h5file:
        episode_keys = sorted(list(h5file.keys()))
        if episode_idx >= len(episode_keys):
            print(f"Episode index {episode_idx} out of range (0-{len(episode_keys)-1})")
            return
        
        ep_key = episode_keys[episode_idx]
        episode = h5file[ep_key]
        
        # Plot joint positions and actions
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot torso positions
        if 'obs/joint_state/torso/joint_position' in episode:
            torso_pos = episode['obs/joint_state/torso/joint_position'][:]
            time = np.arange(len(torso_pos))
            for i in range(torso_pos.shape[1]):
                axes[0].plot(time, torso_pos[:, i], label=f'Joint {i+1}')
            axes[0].set_title('Torso Joint Positions')
            axes[0].set_xlabel('Time Steps')
            axes[0].set_ylabel('Position')
            axes[0].legend()
        
        # Plot arm positions
        if 'obs/joint_state/right_arm/joint_position' in episode:
            right_arm_pos = episode['obs/joint_state/right_arm/joint_position'][:, :6]  # First 6 dimensions
            time = np.arange(len(right_arm_pos))
            for i in range(right_arm_pos.shape[1]):
                axes[1].plot(time, right_arm_pos[:, i], label=f'Joint {i+1}')
            axes[1].set_title('Right Arm Joint Positions')
            axes[1].set_xlabel('Time Steps')
            axes[1].set_ylabel('Position')
            axes[1].legend()
        
        # Plot actions
        if 'action/right_arm' in episode:
            right_arm_action = episode['action/right_arm'][:]
            time = np.arange(len(right_arm_action))
            for i in range(right_arm_action.shape[1]):
                axes[2].plot(time, right_arm_action[:, i], label=f'Joint {i+1}')
            axes[2].set_title('Right Arm Actions')
            axes[2].set_xlabel('Time Steps')
            axes[2].set_ylabel('Action')
            axes[2].legend()
        
        plt.tight_layout()
        plt.savefig('hdf5_episode_visualization.png')
        plt.close()
        
        print(f"HDF5 episode visualization saved to hdf5_episode_visualization.png")
        
        # Show point cloud if available
        if 'obs/point_cloud/fused/xyz' in episode:
            # Use the middle frame of the episode
            mid_idx = len(episode['obs/point_cloud/fused/xyz']) // 2
            
            xyz = episode['obs/point_cloud/fused/xyz'][mid_idx]
            rgb = episode['obs/point_cloud/fused/rgb'][mid_idx]
            
            # Normalize RGB to [0,1] if needed
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
                
            # Sample points for better visualization (max 5000 points)
            max_points = 5000
            if len(xyz) > max_points:
                indices = np.random.choice(len(xyz), max_points, replace=False)
                xyz = xyz[indices]
                rgb = rgb[indices]
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Point Cloud (Episode {episode_idx}, Frame {mid_idx})')
            plt.savefig('hdf5_point_cloud_visualization.png')
            plt.close()
            
            print(f"Point cloud visualization saved to hdf5_point_cloud_visualization.png")

def visualize_zarr_data(zarr_path, episode_idx=0):
    """
    Visualize a sample episode from the Zarr file
    """
    root = zarr.open(zarr_path, 'r')
    
    # Check if data and meta groups exist
    if 'data' not in root or 'meta' not in root:
        print("Invalid Zarr structure. Expected 'data' and 'meta' groups.")
        return
    
    # Get episode boundaries
    episode_ends = root['meta/episode_ends'][:]
    if episode_idx >= len(episode_ends):
        print(f"Episode index {episode_idx} out of range (0-{len(episode_ends)-1})")
        return
    
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx-1]
    end_idx = episode_ends[episode_idx]
    
    # Plot state and action data
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot torso positions (indices 3-7)
    states = root['data/state'][start_idx:end_idx]
    time = np.arange(end_idx - start_idx)
    
    # Torso positions (indices 3-7)
    for i in range(4):
        axes[0].plot(time, states[:, i+3], label=f'Joint {i+1}')
    axes[0].set_title('Torso Joint Positions')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Position')
    axes[0].legend()
    
    # Right arm positions (indices 14-20)
    for i in range(6):
        axes[1].plot(time, states[:, i+14], label=f'Joint {i+1}')
    axes[1].set_title('Right Arm Joint Positions')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Position')
    axes[1].legend()
    
    # Right arm actions (indices 14-20)
    actions = root['data/action'][start_idx:end_idx]
    for i in range(6):
        axes[2].plot(time, actions[:, i+14], label=f'Joint {i+1}')
    axes[2].set_title('Right Arm Actions')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Action')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('zarr_episode_visualization.png')
    plt.close()
    
    print(f"Zarr episode visualization saved to zarr_episode_visualization.png")
    
    # Visualize point cloud if available
    if 'point_cloud' in root['data']:
        # Use the middle frame of the episode
        mid_idx = start_idx + (end_idx - start_idx) // 2
        
        point_cloud = root['data/point_cloud'][mid_idx]
        xyz = point_cloud[:, :3]
        rgb = point_cloud[:, 3:]

        # Normalize RGB to [0,1] if needed
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        # Sample points for better visualization (max 5000 points)
        max_points = 5000
        if len(xyz) > max_points:
            indices = np.random.choice(len(xyz), max_points, replace=False)
            xyz = xyz[indices]
            rgb = rgb[indices]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point Cloud (Episode {episode_idx}, Middle Frame {mid_idx})')
        plt.savefig('zarr_point_cloud_visualization.png')
        plt.close()
        
        print(f"Point cloud visualization saved to zarr_point_cloud_visualization.png")

def compare_datasets(hdf5_path, zarr_path, episode_idx=0):
    """
    Compare an episode from both HDF5 and Zarr datasets
    """
    # Open HDF5
    h5file = h5py.File(hdf5_path, "r", swmr=True, libver="latest")
    episode_keys = sorted(list(h5file.keys()))
    if episode_idx >= len(episode_keys):
        print(f"Episode index {episode_idx} out of range for HDF5 (0-{len(episode_keys)-1})")
        return
    
    ep_key = episode_keys[episode_idx]
    h5_episode = h5file[ep_key]
    
    # Get length
    if 'action/left_arm' in h5_episode:
        h5_length = len(h5_episode['action/left_arm'])
    else:
        print("Missing action/left_arm in HDF5 episode")
        return
    
    # Open Zarr
    root = zarr.open(zarr_path, 'r')
    
    # Check if data and meta groups exist
    if 'data' not in root or 'meta' not in root:
        print("Invalid Zarr structure. Expected 'data' and 'meta' groups.")
        return
    
    # Get episode boundaries
    episode_ends = root['meta/episode_ends'][:]
    if episode_idx >= len(episode_ends):
        print(f"Episode index {episode_idx} out of range for Zarr (0-{len(episode_ends)-1})")
        return
    
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx-1]
    end_idx = episode_ends[episode_idx]
    zarr_length = end_idx - start_idx
    
    # Compare lengths
    print(f"Episode {episode_idx} length comparison:")
    print(f"  HDF5: {h5_length} frames")
    print(f"  Zarr: {zarr_length} frames")
    
    # Early return if lengths don't match
    if h5_length != zarr_length:
        print("WARNING: Episode lengths don't match!")

    
    # Compare left arm actions
    if 'action/right_arm' in h5_episode and 'action' in root['data']:
        h5_right_arm = h5_episode['action/right_arm'][:]
        zarr_actions = root['data/action'][start_idx:end_idx]
        zarr_right_arm = zarr_actions[:, 14:20]  # Indices 7-13 are left arm
        
        min_len = min(len(h5_right_arm), len(zarr_right_arm))

        # Calculate differences
        diff = h5_right_arm[:min_len] - zarr_right_arm[:min_len]
        max_diff = np.max(np.abs(diff))
        mean_diff = np.mean(np.abs(diff))
        
        print(f"Right arm action comparison:")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        
        # Plot comparison
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot original data
        time = np.arange(min_len)
        for i in range(min(h5_right_arm.shape[1], zarr_right_arm.shape[1])):
            axes[0].plot(time, h5_right_arm[:min_len, i], 'b-', label=f'HDF5 Joint {i+1}')
            axes[0].plot(time, zarr_right_arm[:min_len, i], 'r--', label=f'Zarr Joint {i+1}')
        
        axes[0].set_title('Right Arm Actions Comparison')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Action')
        axes[0].legend()
        
        # Plot differences
        for i in range(min(h5_right_arm.shape[1], zarr_right_arm.shape[1])):
            axes[1].set_ylim(5, -5)
            axes[1].plot(time, diff[:, i], label=f'Joint {i+1}')
        
        axes[1].set_title('Differences (HDF5 - Zarr)')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Difference')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('dataset_comparison.png')
        plt.close()
        
        print(f"Dataset comparison visualization saved to dataset_comparison.png")
    else:
        print("No left arm actions found in HDF5 or Zarr")

    # Compare images
    print("\n=== Image Comparison ===")
    image_types = ['head', 'left_wrist', 'right_wrist']
    
    for img_type in image_types:
        h5_path = f'obs/rgb/{img_type}/img'
        zarr_path = f'{img_type}_img'

        if h5_path in h5_episode and zarr_path in root['data']:
            # Get middle frame for comparison
            mid_idx = h5_length // 2
            h5_img = h5_episode[h5_path][mid_idx]
            zarr_img = root['data'][zarr_path][start_idx + mid_idx]
            
            # Calculate differences
            diff = h5_img.astype(np.float32) - zarr_img.astype(np.float32)
            max_diff = np.max(np.abs(diff))
            mean_diff = np.mean(np.abs(diff))
            
            print(f"\n{img_type.replace('_', ' ').title()} image comparison:")
            print(f"  Max absolute difference: {max_diff:.6f}")
            print(f"  Mean absolute difference: {mean_diff:.6f}")
            
            # Plot comparison
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot HDF5 image
            axes[0, 0].imshow(h5_img)
            axes[0, 0].set_title(f'HDF5 {img_type.replace("_", " ").title()} Image')
            axes[0, 0].axis('off')
            
            # Plot Zarr image
            axes[0, 1].imshow(zarr_img)
            axes[0, 1].set_title(f'Zarr {img_type.replace("_", " ").title()} Image')
            axes[0, 1].axis('off')
            
            # Plot difference
            diff_plot = axes[1, 0].imshow(diff, cmap='RdBu', vmin=-255, vmax=255)
            axes[1, 0].set_title('Difference (HDF5 - Zarr)')
            axes[1, 0].axis('off')
            plt.colorbar(diff_plot, ax=axes[1, 0])
            
            # Plot histogram of differences
            axes[1, 1].hist(diff.flatten(), bins=50, range=(-255, 255))
            axes[1, 1].set_title('Difference Histogram')
            axes[1, 1].set_xlabel('Pixel Difference')
            axes[1, 1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(f'image_comparison_{img_type}.png')
            plt.close()
            
            print(f"Image comparison visualization saved to image_comparison_{img_type}.png")
        else:
            print(f"No {img_type} images found in HDF5 or Zarr")

def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 dataset to Zarr format and verify the conversion")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to save Zarr dataset")
    parser.add_argument("--skip_conversion", action="store_true", help="Skip conversion and only verify existing Zarr dataset")
    parser.add_argument("--episode_idx", type=int, default=1, help="Episode index to visualize (default: 0)")    
    args = parser.parse_args()
    
    # Convert dataset if not skipped
    if not args.skip_conversion:
        print("\n=== Converting HDF5 to Zarr ===")
        convert_hdf5_to_zarr(args.hdf5_path, args.zarr_path)
    
    # Visualize HDF5 episode
    print("\n=== Visualizing HDF5 Data ===")
    visualize_hdf5_episode(args.hdf5_path, args.episode_idx)
    
    # Visualize Zarr episode
    print("\n=== Visualizing Zarr Data ===")
    visualize_zarr_data(args.zarr_path, args.episode_idx)
    
    # Compare datasets
    print("\n=== Comparing Datasets ===")
    compare_datasets(args.hdf5_path, args.zarr_path, args.episode_idx)
    
    print("\nVerification complete. Check the generated visualizations.")

if __name__ == "__main__":
    main() 
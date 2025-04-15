#!/usr/bin/env python3

"""
Utility script to convert HDF5 dataset to Zarr format
(mainly for Galaxea R1 robot data from BEHAVIOR SUITE)
"""

import argparse
import numpy as np
import os
import h5py
import zarr
from tqdm import tqdm

def convert_hdf5_to_zarr(hdf5_path, zarr_path):
    """
    Convert HDF5 dataset to Zarr format

    Args:
        hdf5_path: Path to HDF5 file
        zarr_path: Path to save Zarr dataset
    """
    print(f"Converting {hdf5_path} to {zarr_path}")
    
    # Open HDF5 file
    with h5py.File(hdf5_path, "r", swmr=True, libver="latest") as h5file:
        episode_keys = sorted(list(h5file.keys()))
        
        # Create Zarr store
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store, overwrite=True)
        data_group = root.create_group('data')  
        meta_group = root.create_group('meta')
        
        # Get data shape from first episode
        first_ep = h5file[episode_keys[0]]
        
        # Calculate total frames across all episodes
        total_frames = 0
        episode_lengths = []
        for ep_key in episode_keys:
            ep_len = len(h5file[ep_key]['action/mobile_base'])
            total_frames += ep_len
            episode_lengths.append(ep_len)
        
        # Create arrays for data
        # State combines: base_vel, joint positions, gripper positions
        state_shape = (3 + 4 + 6 + 1 + 6 + 1)  # Base + torso + left arm + left gripper + right arm + right gripper
        states = data_group.create_dataset('state', shape=(total_frames, state_shape), 
                                  dtype=np.float32, chunks=(256, state_shape))
        
        # Action
        action_shape = (3 + 4 + 6 + 1 + 6 + 1)  # Base + torso + left arm + left gripper + right arm + right gripper
        actions = data_group.create_dataset('action', shape=(total_frames, action_shape), 
                                   dtype=np.float32, chunks=(256, action_shape))
        
        # Get point cloud shape from first episode
        # Point cloud contains xyz and rgb
        pcd_xyz_shape = h5file[episode_keys[0]]['obs/point_cloud/fused/xyz'].shape
        pcd_shape = (pcd_xyz_shape[1], pcd_xyz_shape[2] + 3)  # xyz + rgb
        
        point_clouds = data_group.create_dataset('point_cloud', shape=(total_frames, pcd_shape[0], pcd_shape[1]),
                                        dtype=np.float32, chunks=(64, pcd_shape[0], pcd_shape[1]))
        
        # Create episode_ends array
        episode_ends = np.cumsum(episode_lengths)
        meta_group.create_dataset('episode_ends', data=episode_ends, dtype=np.int64)
        
        # Copy data
        frame_idx = 0
        for ep_idx, ep_key in enumerate(tqdm(episode_keys)):
            episode = h5file[ep_key]
            ep_len = episode_lengths[ep_idx]
            # State data
            # Base velocity
            states[frame_idx:frame_idx+ep_len, 0:3] = episode['obs/odom/base_velocity'][:]
            # Torso
            states[frame_idx:frame_idx+ep_len, 3:7] = episode['obs/joint_state/torso/joint_position'][:]
            # Left arm (exclude gripper)
            states[frame_idx:frame_idx+ep_len, 7:13] = episode['obs/joint_state/left_arm/joint_position'][:, :-1]
            # Left gripper. 
            # Debug note: the first[:] converts hdpy.Dataset object to numpy array, 
            # the second[:, np.newaxis] converts 1D array to 2D array
            states[frame_idx:frame_idx+ep_len, 13:14] = episode['obs/gripper_state/left_gripper/gripper_position'][:][:, np.newaxis]
            # Right arm (exclude gripper)
            states[frame_idx:frame_idx+ep_len, 14:20] = episode['obs/joint_state/right_arm/joint_position'][:, :-1]
            # Right gripper
            states[frame_idx:frame_idx+ep_len, 20:21] = episode['obs/gripper_state/right_gripper/gripper_position'][:][:, np.newaxis]
            
            # Action data
            actions[frame_idx:frame_idx+ep_len, 0:3] = episode['action/mobile_base'][:]
            actions[frame_idx:frame_idx+ep_len, 3:7] = episode['action/torso'][:]
            actions[frame_idx:frame_idx+ep_len, 7:13] = episode['action/left_arm'][:]
            actions[frame_idx:frame_idx+ep_len, 13:14] = episode['action/left_gripper'][:][:, np.newaxis] / 100.0  # Normalize to 0-1
            actions[frame_idx:frame_idx+ep_len, 14:20] = episode['action/right_arm'][:]
            actions[frame_idx:frame_idx+ep_len, 20:21] = episode['action/right_gripper'][:][:, np.newaxis] / 100.0  # Normalize to 0-1
            
            # Point cloud data
            xyz = episode['obs/point_cloud/fused/xyz'][:]
            rgb = episode['obs/point_cloud/fused/rgb'][:]
            
            for t in range(ep_len):
                # Combine xyz and rgb into single array
                combined = np.concatenate([xyz[t], rgb[t]], axis=1)
                point_clouds[frame_idx+t] = combined
                
            frame_idx += ep_len
    
    print(f"Conversion completed. Zarr dataset saved to {zarr_path}")
    print(f"Total frames: {total_frames}, Total episodes: {len(episode_keys)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 dataset to Zarr format")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to save Zarr dataset")
    
    args = parser.parse_args()
    convert_hdf5_to_zarr(args.hdf5_path, args.zarr_path) 
from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, StringNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import diffusion_policy_3d.model.vision_3d.point_process as point_process
from termcolor import cprint
from scipy.ndimage import zoom

class R1CupDatasetImage(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            use_img=True,
            use_depth=False,
            ):
        super().__init__()
        cprint(f'Loading R1CupDataset from {zarr_path}', 'green')
        self.task_name = task_name
        self.use_img = use_img
        self.use_depth = use_depth

        buffer_keys = [
            'state', 
            'action',]
        
        if self.use_img:
            buffer_keys.extend(['head_img', 'left_wrist_img', 'right_wrist_img'])
        if self.use_depth:
            buffer_keys.append('depth')

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=buffer_keys)
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {'action': self.replay_buffer['action'],
                'agent_pos': self.replay_buffer['state']
                }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        if self.use_img:
            # Create identity normalizers for each image view
            # normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['head_image'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['left_wrist_image'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['right_wrist_image'] = SingleFieldLinearNormalizer.create_identity()
        if self.use_depth:
            normalizer['depth'] = SingleFieldLinearNormalizer.create_identity()
        
        # normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        action = sample['action'][:,].astype(np.float32)

        #NOTE: since normalizer is set before sampling,
        # we need to rectify the gripper action/state BEFORE forming the dataset,
        # not here,
        # otherwise the normalizer parameters will be incorrect

        if self.use_img:
            # Get images from all three cameras
            head_img = sample['head_img'][:,].astype(np.float32)
            left_wrist_img = sample['left_wrist_img'][:,].astype(np.float32)
            right_wrist_img = sample['right_wrist_img'][:,].astype(np.float32)
            
        if self.use_depth:
            depth = sample['depth'][:,].astype(np.float32)
            
        data = {
            'obs': {
                'agent_pos': agent_pos,
                },
            'action': action}
        if self.use_img:
            data['obs']['head_image'] = head_img
            data['obs']['left_wrist_image'] = left_wrist_img
            data['obs']['right_wrist_image'] = right_wrist_img
        if self.use_depth:
            # data['obs']['depth'] = depth 
            raise NotImplementedError("Depth is not used in the model. May need to change this for multi-view")
            
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch_function = lambda x: torch.from_numpy(x) if x.__class__.__name__ == 'ndarray' else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data


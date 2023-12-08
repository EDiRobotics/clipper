import io
import gc
from time import time
import lmdb
from pickle import loads
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg
from torchvision.transforms import ToPILImage


class LMDBDataset(Dataset):
    def __init__(self, lmdb_dir, history_len, chunk_size, desired_rgb_shape, action_space, num_actions):
        super(LMDBDataset).__init__()
        self.history_len = history_len
        self.chunk_size = chunk_size
        self.action_space = action_space
        self.num_actions = num_actions
        self.dummy_timesteps = torch.zeros(history_len, 1, dtype=torch.int)
        self.dummy_rgbs = torch.zeros((history_len, 3,) + desired_rgb_shape)
        self.dummy_actions = torch.zeros(history_len, chunk_size, num_actions)
        self.dummy_distances = torch.zeros(history_len, chunk_size + 1, 1)
        self.lmdb_dir = lmdb_dir
        env = lmdb.open(lmdb_dir, readonly=True, create=False, lock=False)
        with env.begin() as txn:
            self.length = loads(txn.get('cur_step'.encode()))
            episode_num = loads(txn.get(f'cur_episode_{self.length}'.encode()))
            self.avg_len = self.length / episode_num
        env.close()
        self.read_num = 0
        self.read_uplimit = 500 * 16

    def open_lmdb(self):
        '''
        There may be memory leak or just some tensor not free by pytorch
        I have no time to profile this, just periodically collect gabbage...
        Good luck my friend!
        '''
        if hasattr(self, 'env'):
            gc.collect()
            self.txn.abort()
            self.env.close()
        self.env = lmdb.open(self.lmdb_dir, readonly=True, create=False, lock=False)
        self.txn = self.env.begin()

    def __getitem__(self, idx):
        if self.read_num == 0 or self.read_num == self.read_uplimit:
            self.open_lmdb()
            self.read_num = 0
        self.read_num += 1

        cur_episode = loads(self.txn.get(f'cur_episode_{idx}'.encode()))
        inst = loads(self.txn.get(f'inst_{cur_episode}'.encode()))
        episode_start = loads(self.txn.get(f'episode_start_{cur_episode}'.encode()))
        episode_end = loads(self.txn.get(f'episode_end_{cur_episode}'.encode()))

        timesteps = self.dummy_timesteps.clone()
        distances = self.dummy_distances.clone()
        rgbs = self.dummy_rgbs.clone()
        actions = self.dummy_actions.clone()
        for i in range(self.history_len):
            if idx + i > episode_end:
                # The state is in next episode, don't store it
                break
            timesteps[i] = 1 + idx + i - episode_start  # start from 1
            jpg = loads(self.txn.get(f'rgb_{idx + i}'.encode()))
            rgbs[i] = decode_jpeg(jpg)
            distances[i, 0] = (episode_end - idx + i) / self.avg_len
            for k in range(self.chunk_size):
                if idx + i + k > episode_end:
                    # The action is in next episode, don't store it
                    # but the subsequent states may still be in current episode
                    continue
                actions[i, k] = loads(self.txn.get(f'cont_action_{idx + i + k}'.encode()))
                distances[i, k + 1] = (episode_end - (idx + i + k + 1)) / self.avg_len

        actions = (actions - self.action_space[0]) / (self.action_space[1] - self.action_space[0])
        return rgbs, inst, actions, timesteps, distances

    def __len__(self):
        return self.length - self.history_len - self.chunk_size + 1


class CLIP_LMDBDataset(Dataset):
    def __init__(self, path, preprocess, tokenizer,
                 history_len=1,
                 chunk_size=1,
                 desired_rgb_shape=(192, 320),
                 action_space=[-0.06, 0.06],
                 num_actions=2
                 ):
        lmdb_dataset = LMDBDataset(
            path,
            history_len=history_len,
            chunk_size=chunk_size,
            desired_rgb_shape=desired_rgb_shape,
            action_space=action_space,
            num_actions=num_actions,
        )
        self.lmdb_dataset = lmdb_dataset
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.lmdb_dataset)

    def __getitem__(self, idx):
        image, text, _, _, _ = self.lmdb_dataset[idx]
        to_pil = ToPILImage()
        image_pil = to_pil(image.squeeze(0))
        image = self.preprocess(image_pil)
        text = self.tokenizer(text).squeeze(0)

        return image, text


if __name__ == "__main__":
    path = '/Downloads/Dataset_1201_rgb_small'
    dataset = LMDBDataset(
        path,
        history_len=1,
        chunk_size=10,
        desired_rgb_shape=(192, 320),
        action_space=[-0.06, 0.06],
        num_actions=2,
    )
    data = dataset[97]
    print(data)
    import pdb;

    pdb.set_trace()

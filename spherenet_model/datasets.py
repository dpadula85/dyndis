#!/usr/bin/env python

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.utils import shuffle

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset


class DimersSP2NoH(InMemoryDataset):
    def __init__(self, root='couplings_sp2_noh/', transform=None, pre_transform=None, pre_filter=None):

        self.folder = os.path.join(root, 'data')
        super(DimersSP2NoH, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data_torch_sp2.npz'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        
        data = np.load(os.path.join(self.raw_dir, self.raw_file_names), allow_pickle=True)
        N = data['N']
        R = data['R']
        Z = data['Z']
        mol = list(map(lambda x: x.split(os.sep)[0].upper(), data["files"]))
        didx = list(map(lambda x: os.path.split(x)[-1].split("_")[1], data["files"]))
        didx = list(map(int, didx))
        aidx = list(map(lambda x: os.path.split(x)[-1].split("_")[2], data["files"]))
        aidx = list(map(int, aidx))
        time = list(map(lambda x: os.path.split(x)[-1].split("_")[3].split(".")[0], data["files"]))
        time = list(map(float, time))
        split = np.cumsum(N)
        R_dim = np.split(R, split)
        Z_dim = np.split(Z, split)
        target = {}
        target["Y"] = np.expand_dims(data["Y"],axis=-1)

        data_list = []
        for i in tqdm(range(len(N))):
            R_i = torch.tensor(R_dim[i],dtype=torch.float32)
            z_i = torch.tensor(Z_dim[i],dtype=torch.int64)
            y_i = torch.tensor(target["Y"][i],dtype=torch.float32)
            mol_i = mol[i]
            didx_i = didx[i]
            aidx_i = aidx[i]
            time_i = time[i]
            data = Data(
                    pos=R_i,
                    z=z_i,
                    y=y_i,
                    mol=mol_i,
                    t=time_i,
                    didx=didx_i,
                    aidx=aidx_i,
                    shift="0",
                )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [ data for data in data_list if self.pre_filter(data) ]
        if self.pre_transform is not None:
            data_list = [ self.pre_transform(data) for data in data_list ]
        
        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict


class GradDimersSP2NoH(InMemoryDataset):
    def __init__(self, root='gradients_sp2_noh/', transform=None, pre_transform=None, pre_filter=None):

        self.folder = os.path.join(root, 'data')
        super(GradDimersSP2NoH, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'grads_torch_sp2.npz'

    @property
    def processed_file_names(self):
        return 'grads_data.pt'

    def process(self):
        
        data = np.load(os.path.join(self.raw_dir, self.raw_file_names), allow_pickle=True)
        N = data['N']
        R = data['R']
        Z = data['Z']
        mol = list(map(lambda x: x.split(os.sep)[0].upper(), data["files"]))
        didx = list(map(lambda x: os.path.split(x)[-1].split("_")[1], data["files"]))
        didx = list(map(int, didx))
        aidx = list(map(lambda x: os.path.split(x)[-1].split("_")[2], data["files"]))
        aidx = list(map(int, aidx))
        time = list(map(lambda x: os.path.split(x)[-1].split("_")[3], data["files"]))
        time = list(map(float, time))
        shift = list(map(lambda x: os.path.split(x)[-1].split("_")[4].split(".")[0], data["files"]))
        split = np.cumsum(N)
        R_dim = np.split(R, split)
        Z_dim = np.split(Z, split)
        target = {}
        target["Y"] = np.expand_dims(data["Y"],axis=-1)

        data_list = []
        for i in tqdm(range(len(N))):
            R_i = torch.tensor(R_dim[i],dtype=torch.float32)
            z_i = torch.tensor(Z_dim[i],dtype=torch.int64)
            y_i = torch.tensor(target["Y"][i],dtype=torch.float32)
            mol_i = mol[i]
            didx_i = didx[i]
            aidx_i = aidx[i]
            time_i = time[i]
            shift_i=shift[i]
            data = Data(
                    pos=R_i,
                    z=z_i,
                    y=y_i,
                    mol=mol_i,
                    t=time_i,
                    didx=didx_i,
                    aidx=aidx_i,
                    shift=shift_i,
                )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [ data for data in data_list if self.pre_filter(data) ]
        if self.pre_transform is not None:
            data_list = [ self.pre_transform(data) for data in data_list ]
        
        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict


if __name__ == '__main__':

    dataset = DimersSP2NoH()

    try:
        split_idx = torch.load("splits_sp2.pt")
    except:
        split_idx = dataset.get_idx_split(len(dataset._data.y), train_size=21000, valid_size=7100, seed=42)
        torch.save(split_idx, "splits_sp2.pt")

    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    # Gradients
    dataset = GradDimersSP2NoH()

    try:
        split_idx = torch.load("grads_splits_sp2.pt")
    except:
        split_idx = dataset.get_idx_split(len(dataset._data.y), train_size=7800, valid_size=2500, seed=42)
        torch.save(split_idx, "grads_splits_sp2.pt")

    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

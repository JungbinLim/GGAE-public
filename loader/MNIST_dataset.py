######################################################
#                                                    #
# code modified from IRVAE_public (Lee et al., 2022) #
#                                                    #
######################################################

import numpy as np
import torch
import os, sys
from torchvision.datasets.mnist import MNIST as MNIST_torch

sys.path.append('..')
from kernels import get_distfunc

class MNIST(MNIST_torch):
    def __init__(self,
        root,
        digits=[0,1],
        download=True,
        split='training',
        **kwargs):

        super(MNIST, self).__init__(
            root,
            download=download,
        )
        
        verbose = kwargs.get('verbose', True)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")

        self.train = True
        data1, targets1 = self._load_data()
        self.train = False
        data2, targets2 = self._load_data()

        data = (torch.cat([data1, data2], dim=0).to(torch.float32) / 255).unsqueeze(1)
        targets = torch.cat([targets1, targets2], dim=0)

        # use only a fraction of data when computing global distance matrix (too heavy)
        assert len(data)==len(targets)
        
        fraction_data = kwargs.get("fraction_data", None)
        if verbose:
            print("fraction_data:", fraction_data)
        if fraction_data is not None:
            fraction_len = round(fraction_data*data.shape[0])
            data = data[:fraction_len]
            targets = targets[:fraction_len]
            assert len(data)==len(targets)
        
        if isinstance(digits, str):
            if digits.startswith('list'):
                digits = [int(i) for i in digits.split('_')[1]] 
            elif digits == 'all':
                pass
            else:
                raise ValueError

        if digits == "all":
            pass
        else:
            data_list = []
            targets_list = []
            for d, t in zip(data, targets):
                if t in digits:
                    data_list.append(d.unsqueeze(0))
                    targets_list.append(t.unsqueeze(0))
            data = torch.cat(data_list, dim=0)
            targets = torch.cat(targets_list, dim=0)

        split_train_val_test = (5/7, 1/7, 1/7)
        num_train_data = int(len(data) * split_train_val_test[0])
        num_valid_data = int(len(data) * split_train_val_test[1]) 

        if split == "training":
            data = data[:num_train_data]
            targets = targets[:num_train_data]
        elif split == "validation":
            data = data[num_train_data:num_train_data + num_valid_data]
            targets = targets[num_train_data:num_train_data + num_valid_data]
        elif split == "test":
            data = data[num_train_data + num_valid_data:]
            targets = targets[num_train_data + num_valid_data:]

        self.data = data            # (ttl, *dims)
        self.targets = targets      # (ttl)
        
        print(f"MNIST split {split} | {len(self.data)}")
        
        self.precompute_global_dist = kwargs.get("precompute_global_dist", False)
        self.global_dist = None
        if self.precompute_global_dist:
            distfunc = kwargs.get("distfunc", "Euclidean")
            X = self.data.unsqueeze(0).flatten(2,-1)                # (1, ttl, d)
            Y = self.targets.unsqueeze(0).unsqueeze(-1)             # (1, ttl, 1)
            dist = get_distfunc(name=distfunc, data_cfg = kwargs)   # (1, ttl, 1)
            self.global_dist = dist(xA=X, xB=X, yA=Y, yB=Y)
            if verbose:
                print("global_dist shape:", self.global_dist.shape)
        else:
            if verbose:
                print("no global Laplacian, normalized K, or distance matrix is precomputed")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y, idx
    
class RotatingMNIST:
    # Use pre-generated RotatingMNIST dataset file
    def __init__(
        self, 
        root, 
        digits=[0,1], 
        # download=True, 
        split='training', 
        time_horizon=36,
        **kwargs):

        verbose = kwargs.get('verbose', True)
        
        if digits == "all":
            digits = list(range(10))
            
        split_train_val_test = (5/7, 1/7, 1/7)
        
        data = []
        targets = []
        for digit in digits:
            file_name = f'RotatingMNIST-digit={digit}.pkl'
            try:
                print(os.path.join(root, 'RotatingMNIST', file_name))
                data_dict = torch.load(os.path.join(root, 'RotatingMNIST', file_name))
            except FileNotFoundError:
                raise RuntimeError("Dataset not found.")
            
            N_data = len(data_dict['data'])
            num_train_data = int(N_data * split_train_val_test[0])
            num_valid_data = int(N_data * split_train_val_test[1]) 
            
            idx = np.arange(N_data)
            train_idx = idx[:num_train_data]
            valid_idx = idx[num_train_data:num_train_data + num_valid_data]
            test_idx = idx[num_train_data + num_valid_data:]
            
            assert time_horizon <= data_dict['data'].shape[-1], f'Possible maximum time horizon length is {data_dict["data"].shape[-1]}, but requested {time_horizon}.'
            
            if split == "training":
                data.append(data_dict['data'][train_idx][..., :time_horizon])
                targets.append(data_dict['targets'][train_idx])
            elif split == "validation":
                data.append(data_dict['data'][valid_idx][..., :time_horizon])
                targets.append(data_dict['targets'][valid_idx])
            elif split == "test":
                data.append(data_dict['data'][test_idx][..., :time_horizon])
                targets.append(data_dict['targets'][test_idx])
            elif split == "all":
                data.append(data_dict['data'][..., :time_horizon])
                targets.append(data_dict['targets'])
            
        data = torch.cat(data, dim=0)
        targets = torch.cat(targets, dim=0)

        fraction_data = kwargs.get("fraction_data", None)
        if verbose:
            print("fraction_data:", fraction_data)
        if fraction_data is not None:
            fraction_len = round(fraction_data*data.shape[0])
            data = data[:fraction_len]
            targets = targets[:fraction_len]
            if verbose:
                print("fraction:", data.shape, targets.shape)    
            assert len(data)==len(targets)
        
        self.data = data
        self.targets = targets

        if verbose:
            print(f"RotatingMNIST split {split} | {self.data.size()}")
        
        self.global_dist = None
        self.precompute_global_dist = kwargs.get("precompute_global_dist", False)
        if self.precompute_global_dist:
            raise NotImplementedError       # Graph is defined on each time sequence, so it is not possible to compute global distance matrix.
        else:
            if verbose:
                print("no global_dist is precomputed")
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y, idx
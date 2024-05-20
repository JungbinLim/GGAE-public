import numpy as np
import torch

from scipy.optimize import fsolve

from kernels import get_distfunc

class SwissRoll:
    def __init__(self, N=10000, uniform=False, hole=True, theta_range = [1.5,3.5], hole_range = [0.33, 0.67, 0.33, 0.67], aspect_ratio = 1, split='training', **kwargs):
        
        verbose = kwargs.get('verbose', True)
        
        th0 = torch.tensor(np.pi * theta_range[0])
        thf = torch.tensor(np.pi * theta_range[1])
        
        s0 = self._th2s(th0)
        sf = self._th2s(thf)
        
        z0 = 0
        zf = sf - s0

        # If hole==True, set enough number to sample
        if hole:
            N_temp = int(2 * N / (1 - (hole_range[1] - hole_range[0]) * (hole_range[3] - hole_range[2])))       # factor of 2 for enough non-hole points
        else:
            N_temp = N
        
        # Smaple uniformly or non-uniformly
        if uniform:
            s = s0 + (sf - s0) * torch.linspace(0,1,int(np.sqrt(N_temp))).repeat_interleave(int(np.sqrt(N_temp)))
            z = z0 + (zf - z0) * torch.linspace(0,1,int(np.sqrt(N_temp))).repeat(int(np.sqrt(N_temp)),1).flatten(0,1)
        else:
            s = s0 + (sf - s0) * torch.rand(N_temp)
            z = z0 + (zf - z0) * torch.rand(N_temp)  
        
        # If hole==True, only use non-hole points
        if hole:
            temp = torch.logical_and(torch.logical_and(s < s0 + (sf - s0) * hole_range[1], s > s0 + (sf - s0) * hole_range[0]),
                                    torch.logical_and(z < z0 + (zf - z0) * hole_range[3], z > z0 + (zf - z0) * hole_range[2]))
            s = s[~temp]
            z = z[~temp]
        else:
            pass
        
        # reset N to the number of obtained points
        N = len(s)
            
        color = (s - s0) / (sf - s0) - 0.5
        
        th = torch.zeros(N)
        for i in range(N):
            th[i] = self._s2th(s[i])
            
        data = torch.zeros(N, 3)
        data[:, 0] = th * torch.cos(th)
        data[:, 1] = th * torch.sin(th)
        data[:, 2] = z
        
        targets = color
        
        split_train_val_test = (5/7, 1/7, 1/7)
        
        num_train_data = int(N * split_train_val_test[0])
        num_valid_data = int(N * split_train_val_test[1]) 
        
        idx = np.random.permutation(N)
        train_idx = idx[:num_train_data]
        valid_idx = idx[num_train_data:num_train_data + num_valid_data]
        test_idx = idx[num_train_data + num_valid_data:]
        
        if split == "training":
            self.data = data[train_idx]
            self.targets = targets[train_idx]
            self.s = s[train_idx]
            
        elif split == "validation":
            self.data = data[valid_idx]
            self.targets = targets[valid_idx]
            self.s = s[valid_idx]
            
        elif split == "test":
            self.data = data[test_idx]
            self.targets = targets[test_idx]
            self.s = s[test_idx]
            
        elif split == "all":
            self.data = data[idx]
            self.targets = targets[idx]
            self.s = s[idx]
        
        if verbose:
            print(f"{split} split,", len(self.data))
        
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
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y, idx
    
    def _s2th(self, s):
        s = s.item()
        y = lambda th: 0.5 * (th * np.sqrt(th**2 + 1) + np.log(th + np.sqrt(th**2 + 1))) - s
        th = fsolve(y, 1)
        return torch.tensor(th)
    
    def _th2s(self, th):
        s = 0.5 * (th * torch.sqrt(th**2 + 1) + torch.log(th + torch.sqrt(th**2 + 1)))
        return s
import numpy as np
import torch
import os, sys

from kernels import get_distfunc

class dSprites:
    def __init__(self, root='dataset', N=10000, split='training', shape='all', fix_scale=False, fix_orientation=True, XY_hole=True, **kwargs):
        verbose = kwargs.get('verbose', True)
        
        dataset_path = os.path.join(root, 'dsprites', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        try:
            dataset = np.load(dataset_path)
        except FileNotFoundError:
            print(f'Fail to load the data file: {dataset_path}')
        
        imgs = dataset['imgs']
        
        latents_values = dataset['latents_values']
        
        ''' 
        imgs : numpy.array, (737280, 64, 64)
        
        latent values
            0: Color, White(1) only
            1: Shape, Square(1), Ellipse(2), Heart(3)       # becomes 0-th axis when no color
            2: Scale, np.linspace(0.5, 1, 6)                # becomes 1-th axis when no color
            3: Orientation, np.linspace(0, 2*np.pi, 40)     # becomes 2-th axis when no color
            4: PosX, np.linspace(0, 1, 32)                  # becomes 3-th axis when no color
            5: PosY, np.linspace(0, 1, 32)                  # becomes 4-th axis when no color
        '''
        
        latents_sizes = np.array([1, 3, 6, 40, 32, 32])
        
        if shape == 'all':
            pass
        else:
            latents_sizes[1] = 1
            if shape == 'square':
                imgs = imgs[latents_values[:,1] == 1]
                latents_values = latents_values[latents_values[:,1] == 1]
            elif shape == 'ellipse':
                imgs = imgs[latents_values[:,1] == 2]
                latents_values = latents_values[latents_values[:,1] == 2]
            elif shape == 'heart':
                imgs = imgs[latents_values[:,1] == 3]
                latents_values = latents_values[latents_values[:,1] == 3]
            else:
                print("shape must be one of ['all', 'square', 'ellipse', 'heart']")
                raise NotImplementedError
        
        if fix_scale:
            latents_sizes[2] = 1
            imgs = imgs[latents_values[:,2] == 1]
            latents_values = latents_values[latents_values[:,2] == 1]
        
        if fix_orientation:
            latents_sizes[3] = 1
            imgs = imgs[latents_values[:,3] == 0]
            latents_values = latents_values[latents_values[:,3] == 0]
            
        if N > len(imgs):
            if verbose:
                print(f"Warning: N ({N}) is larger than the number of images ({len(imgs)}). N is set to {len(imgs)}")
            N = len(imgs)
            sampled_indices = np.random.permutation(N)
        else:
            latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))
            samples = np.zeros((N, latents_sizes.size))
            for lat_i, lat_size in enumerate(latents_sizes):
                samples[:, lat_i] = np.random.randint(lat_size, size=N)
            sampled_indices = np.dot(samples, latents_bases).astype(int)
            
        data = torch.tensor(imgs[sampled_indices], dtype=torch.float).unsqueeze(1)      # (737280, 1, 64, 64)
        targets = torch.tensor(latents_values[sampled_indices, 1:], dtype=torch.float)  # except the color
        
        if XY_hole:
            hole_range = [0.333, 0.666]
            non_hole_idx = ((targets[:, 3] < hole_range[0]) | (targets[:, 3] > hole_range[1])) | ((targets[:, 4] < hole_range[0]) | (targets[:, 4] > hole_range[1]))
            data = data[non_hole_idx]
            targets = targets[non_hole_idx]
            N = len(data)
        
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
        elif split == "validation":
            self.data = data[valid_idx]
            self.targets = targets[valid_idx]
        elif split == "test":
            self.data = data[test_idx]
            self.targets = targets[test_idx]
        elif split == "all":
            self.data = data
            self.targets = targets

        self.precompute_global_dist = kwargs.get("precompute_global_dist", False)
        self.global_dist = None
        if self.precompute_global_dist:
            distfunc = kwargs.get("distfunc", "Euclidean_large")
            X = self.data.unsqueeze(0).flatten(2,-1)                # (1, ttl, d)
            Y = self.targets.unsqueeze(0)                           # (1, ttl, 5)
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
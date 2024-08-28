import torch
from geometry import get_laplacian

class Laplacian_collate_fn(object):
    def __init__(self, data_dict, device, **kwargs):
        self.data_dict = data_dict
        self.device = device
        self.precompute_global_dist = kwargs.get('precompute_global_dist', False)
        self.global_dist = kwargs.get('global_dist', None)
    
    def __call__(self, batch):
        # ------ inputs ------
        # batch = B tuples of (x, y, idx)
        # self.global_laplacian:    (1, ttl, ttl) if self.precompute_global_laplacian is True
        # self.global_dist:         (1, ttl, ttl) if self.precompute_global_dist is True
        
        # ------ outputs ------
        # X: 1 x B x dims   (dims=[C,W,H] for img data)
        # Y: 1 x B
        # L: 1 x B x B      (Laplacian of all the data points in the batch)
        
        distfunc = self.data_dict.get('distfunc', 'Euclidean')
        bandwidth = self.data_dict.get('bandwidth', 50)
        data_list, label_list, idx_list = [], [], []
        for _data, _label, _idx in batch:
            data_list.append(_data)     # _data.shape = (*dims)
            label_list.append(_label)   # _label.shape = (torch.Size([])
            idx_list.append(_idx)
        
        device = self.device
        X = torch.stack(data_list).to(device)   # (B, *dims)
        Y = torch.stack(label_list).to(device)  # (B, label_dims)
        
        X_reshaped = X.reshape(1, len(batch), -1)
        Y_reshaped = Y.reshape(1, len(batch), -1)
        
        precomputed_dist = None
        if self.precompute_global_dist:
            self.global_dist = (self.global_dist).to(device)
            precomputed_dist = self.global_dist[:,idx_list,:][:,:,idx_list]
        
        L = get_laplacian(
            X=X_reshaped,
            Y=Y_reshaped, 
            distfunc=distfunc,
            bandwidth=bandwidth,
            data_cfg = self.data_dict,
            precomputed_dist = precomputed_dist
        )
            
        return X, Y, L

class TimeSeries_collate_fn(object):
    def __init__(self, data_dict, device, **kwargs):
        self.device = device
    
    def __call__(self, batch):
        # ------ inputs ------
        # batch = B tuples of (x, y)
        
        # ------ outputs ------
        # X: (B,T,*dims)   (dims=[C,W,H] for img data)
        # Y: (B,T)
        # L: (B,T,T)      (Laplacian of all the data points in the batch)
        
        data_list, label_list, idx_list = [], [], []
        for _data, _label, _idx in batch:      # _idx added in __getitem__() of dataset class
            data_list.append(_data)
            label_list.append(_label)
            idx_list.append(_idx)

        device = self.device
        X = torch.stack(data_list).to(device)               # [B, *dims, T]
        Y = torch.stack(label_list).to(device)              # [B] or [B, d, T]
        idx = torch.tensor(idx_list).to(device)             # [B]
        
        X = X.unsqueeze(1).transpose(1,-1).squeeze(-1)      # (B,T,*dims)
        T = X.shape[1]
        if len(Y.shape) == 1:
            Y = Y.repeat_interleave(T).view(len(batch),T)   # (B,T)
        else:
            Y = Y.transpose(1, 2)
        
        return X, Y, idx
    
class TimeSeries_dl_collate_fn(object):
    def __init__(self, data_dict, device, **kwargs):
        self.device = device
    
    def __call__(self, batch):
        # ------ inputs ------
        # batch = B tuples of (x, y)
        
        # ------ outputs ------
        # X: (B,*dims,T)   (dims=[C,W,H] for img data)
        # Y: (B,T)
        # L: (B,T,T)      (Laplacian of all the data points in the batch)
        
        data_list, label_list, idx_list = [], [], []
        for _data, _label, _idx in batch:      # _idx added in __getitem__() of dataset class
            data_list.append(_data)
            label_list.append(_label)
            idx_list.append(_idx)

        device = self.device
        X = torch.stack(data_list).to(device)               # [B, *dims, T]
        Y = torch.stack(label_list).to(device)              # [B] or [B, d, T]
        idx = torch.tensor(idx_list).to(device)             # [B]
        
        T = X.shape[-1]
        if len(Y.shape) == 1:
            Y = Y.repeat_interleave(T).view(len(batch),T)   # (B,T)
        else:
            Y = Y.transpose(1, 2)
        
        return X, Y, idx

class TimeSeries_Laplacian_collate_fn(object):
    def __init__(self, data_dict, device, **kwargs):
        self.data_dict = data_dict
        self.device = device
    
    def __call__(self, batch):
        # ------ inputs ------
        # batch = B tuples of (x, y)
        
        # ------ outputs ------
        # X: (B,T,*dims)   (dims=[C,W,H] for img data)
        # Y: (B,T)
        # L: (B,T,T)      (Laplacian of all the data points in the batch)
        
        distfunc = self.data_dict.get('distfunc', 'Euclidean')
        bandwidth = self.data_dict.get('bandwidth', 50)
        
        data_list, label_list, idx_list = [], [], []
        for _data, _label, _idx in batch:      # _idx added in __getitem__() of dataset class
            data_list.append(_data)
            label_list.append(_label)
            idx_list.append(_idx)

        device = self.device
        X = torch.stack(data_list).to(device)               # [batch_size, *dims, T]
        Y = torch.stack(label_list).to(device)              # [batch_size] or [batch_size, d, T]
        
        X = X.unsqueeze(1).transpose(1,-1).squeeze(-1)      # (B,T,*dims)
        T = X.shape[1]
        if len(Y.shape) == 1:
            Y = Y.repeat_interleave(T).view(len(batch),T)       # (B,T)
        else:
            Y = Y.transpose(1, 2)
        
        # reshape X and Y to input into get_laplacian
        X_reshaped = X.reshape(len(batch),T,-1)                # (B,T,d)
        Y_reshaped = Y.reshape(len(batch),T,-1)                # (B,T,1)
        
        # get_laplacian needs x=(B, T, d), y=(B,T,1) and outputs L=(B,T,T)
        L = get_laplacian(
            X_reshaped,
            Y_reshaped,
            distfunc=distfunc,
            bandwidth=bandwidth,
            data_cfg = self.data_dict
        )
        
        return X, Y, L
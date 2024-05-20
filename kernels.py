import torch
import torch.nn.functional as f
import numpy as np
import time
from tqdm import tqdm, trange

from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def get_distfunc(**kwargs):
    name = kwargs['name']
    return {
        'Euclidean': Euclidean,
        'Euclidean_large': Euclidean_large,  
        'Euclidean_knn': Euclidean_knn,
        
        'dSprites_latent': dSprites_latent,
        'dSprites_latent_knn_grid': dSprites_latent_knn_grid,
        
        'SwissRoll_geodesic': SwissRoll_geodesic,
        'TimeSteps': TimeSteps,
    }[name](**kwargs)


class DistFunc:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError('You need to define your own __call__ function.')
    
    def _repeat(self, A, B):
        assert len(A) == len(B), f'Batch size mismatch: got {len(A)} and {len(B)}'
        # assert A.shape[2] == B.shape[2], f'vector dim mismatch: got {A.shape[2]} and {B.shape[2]}'
        
        nBatch = len(A)
        nA, nB, dim = A.shape[1], B.shape[1], A.shape[2]
        
        catA = A.repeat_interleave(nB, dim=1)
        catB = B.repeat(1, nA, 1)
        
        return catA, catB, nBatch, nA, nB, dim
    
class Euclidean(DistFunc):
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, xA, xB, **kwargs):
        # ------ inputs -------
        # A,B : (B, N, d) data
        
        # ------ outputs -------
        # dist : (B, N, N)
        
        xA = xA.unsqueeze(2)                      # (B,N,1,d)
        xB = xB.unsqueeze(1)                      # (B,1,N,d)
        dist = torch.norm(xA-xB, dim=3, p=2)      # subtraction A-B automatically broadcasts
        
        return dist
    
class Euclidean_large(DistFunc):
    def __init__(self, **kwargs):
        data_cfg = kwargs.get("data_cfg", {})
        self.device = data_cfg.get("distfunc_device", 'cpu')
        self.verbose = data_cfg.get("verbose", False)
    
    def __call__(self, xA, xB, **kwargs):
        # ------ inputs -------
        # xA : (B, N, *dims) 
        # xB : (B, M, *dims) 
        
        # ------ outputs -------
        # dist : (B, N, M)

        bA, N, dA = xA.shape
        bB, M, dB = xB.shape
        original_device = xA.device
        
        assert xA.device == xB.device, f'device mismatch: got {xA.device} and {xB.device}'
        assert bA == bB, f'Batch size mismatch: got {bA} and {bB}'
        assert dA == dB, f'vector dim mismatch: got {dA} and {dB}'
        
        xA = xA.unsqueeze(2)                      # (B,N,1,d)
        xB = xB.unsqueeze(1)                      # (B,1,M,d)
        
        dists = torch.zeros([bA, N, M]).to(self.device)
        
        for idx in trange(dA, disable=not self.verbose):
            dists += (xA[:,:,:,idx].to(self.device) - xB[:,:,:,idx].to(self.device))**2
            
        return torch.sqrt(dists).to(original_device)

class Euclidean_knn(DistFunc):
    # Shortest path distance on Euclidean kNN graph with k=self.k
    def __init__(self, **kwargs):
        self.data_cfg = kwargs.get("data_cfg", None)
        self.k = self.data_cfg.get("k", 5)
        self.limit = self.data_cfg.get("limit", 100.0)
    
    def __call__(self, xA, xB, **kwargs):
        # ------ inputs -------
        # A,B : (B, N, *dims)       # (B, T, *dims) for RotatingMNIST, (1, B, B) for MNIST
        
        # ------ outputs -------
        # dist : (B, N, N)
    
        A_ = xA.cpu().numpy()
        dist_matrix_lst = []
        for i in A_:
            G = kneighbors_graph(X=i, n_neighbors=self.k, mode='distance', include_self=False) 
            graph = csr_matrix(G)
            dist_matrix = dijkstra(csgraph=graph, directed=False, limit=self.limit)
            dist_matrix = torch.Tensor(dist_matrix)
            
            disconnected_ratio = (dist_matrix == np.inf).float().mean()
            print("disconnected_ratio=", disconnected_ratio)
            dist_matrix_lst.append(dist_matrix)
        dist = torch.stack(dist_matrix_lst).to(xA)
        dist = dist.nan_to_num(posinf=self.limit)
        
        return dist

class dSprites_latent(DistFunc):
    # GT 2-norm btwn latent points
    
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, xA, xB, yA, yB, **kwargs):
        # ------ inputs -------
        # xA, XB: data, (B, N, *dims)
        # yA, yB : targets, (B, N, 5)   shape, scale, ori, posX, posY
        
        # ------ outputs -------
        # dist : (B, N, N)
        
        assert yA.shape == yB.shape, 'The shapes of latent vectors (targets) should be the same.'
        assert yA.shape[2] == 5, f'The dim. of latent vectors is 5, got {yA.shape[2]}'
        assert xA.shape[:2] == yA.shape[:2], f'Batch and number of graph nodes of x and y should be the same, got {xA.shape[:2]} and {yA.shape[:2]}'
        B = yA.shape[0]
        N = yA.shape[1]
        
        yA_shape_onehot = f.one_hot((yA[:, :, 0]-1).to(torch.int64), num_classes=3).to(torch.float).unsqueeze(2) # (B, N, 1, 3)
        yB_shape_onehot = f.one_hot((yB[:, :, 0]-1).to(torch.int64), num_classes=3).to(torch.float).unsqueeze(1) # (B, 1, N, 3)
        dist_shape = torch.norm(yA_shape_onehot-yB_shape_onehot, dim=3, p=2)
        
        yA_scale = yA[:, :, [1]].unsqueeze(2) # (B, N, 1, 1)
        yB_scale = yB[:, :, [1]].unsqueeze(1) # (B, 1, N, 1)
        dist_scale = torch.norm(yA_scale-yB_scale, dim=3, p=2)
        
        yA_ori = torch.cat([torch.cos(yA[:, :, [2]]), torch.sin(yA[:, :, [2]])], dim=-1).unsqueeze(2) # (B, N, 1, 2)
        yB_ori = torch.cat([torch.cos(yB[:, :, [2]]), torch.sin(yB[:, :, [2]])], dim=-1).unsqueeze(1) # (B, 1, N, 2)
        dist_ori = torch.norm(yA_ori-yB_ori, dim=3, p=2)
        
        yA_pos = yA[:, :, [3, 4]].unsqueeze(2) # (B, N, 1, 2)
        yB_pos = yB[:, :, [3, 4]].unsqueeze(1) # (B, 1, N, 2)
        dist_pos = torch.norm(yA_pos-yB_pos, dim=3, p=2)

        dist = torch.sqrt(dist_shape**2 + dist_scale**2 + dist_ori**2 + dist_pos**2)        # fixed to be the extended 2-norm
        
        return dist
    
class dSprites_latent_knn_grid(DistFunc):
    def __init__(self, **kwargs):
        self.data_cfg = kwargs.get("data_cfg", None)
        self.k = self.data_cfg.get("k", 5)
        self.limit = self.data_cfg.get("limit", 100.0)
    
    def __call__(self, xA, xB, yA, yB, **kwargs):
        # ------ inputs -------
        # xA, xB: data, (B, N, *dims)
        # yA, yB : targets, (B, N, 5)
        
        # ------ outputs -------
        # dist : (B, N, N)
        
        assert yA.shape == yB.shape, 'The shapes of latent vectors (targets) should be the same.'
        assert yA.shape[2] == 5, f'The dim. of latent vectors is 5, got {yA.shape[2]}'
        assert xA.shape[:2] == yA.shape[:2], f'Batch and number of graph nodes of x and y should be the same, got {xA.shape[:2]} and {yA.shape[:2]}'
        # assert yA == yB, 'knn distance function only works for the same data yA == yB == Y'
        
        B = yA.shape[0]
        N = yA.shape[1]
        
        pos_step = 0.0323       # unit size of xy grid
        scale_step = 0.1        # unit size of scale grid
        
        yA_shape_onehot = f.one_hot((yA[:, :, 0]-1).to(torch.int64), num_classes=3).to(torch.float) # (B, N, 3)
        yA_scale = yA[:, :, [1]] # (B, N, 1)
        yA_ori = torch.cat([torch.cos(yA[:, :, [2]]), torch.sin(yA[:, :, [2]])], dim=-1) # (B, N, 2)
        yA_posX = yA[:, :, [3]] # (B, N, 1)
        yA_posY = yA[:, :, [4]] # (B, N, 1)
        
        Y_extended_grid = torch.cat([yA_shape_onehot, yA_scale/scale_step, yA_ori, yA_posX/pos_step, yA_posY/pos_step], dim=2) # (B, N, 8)
        Y_extended = torch.cat([yA_shape_onehot, yA_scale, yA_ori, yA_posX, yA_posY], dim=2) # (B, N, 8)
        
        dist_shape = torch.norm(yA_shape_onehot.unsqueeze(2)-yA_shape_onehot.unsqueeze(1), dim=3, p=2)
        dist_scale = torch.norm(yA_scale.unsqueeze(2)-yA_scale.unsqueeze(1), dim=3, p=2)
        dist_ori = torch.norm(yA_ori.unsqueeze(2)-yA_ori.unsqueeze(1), dim=3, p=2)
        dist_posX = torch.norm(yA_posX.unsqueeze(2)-yA_posX.unsqueeze(1), dim=3, p=2)
        dist_posY = torch.norm(yA_posY.unsqueeze(2)-yA_posY.unsqueeze(1), dim=3, p=2)
        dist_grid = dist_shape + dist_scale + dist_ori + dist_posX + dist_posY
        
        A_grid = Y_extended_grid.cpu().numpy()
        dist_grid_ = dist_grid.cpu().numpy()
        dist_matrix_lst = []
        for i_grid, d in zip(A_grid, dist_grid_):
            G_grid = kneighbors_graph(X=i_grid, n_neighbors=self.k, mode='connectivity', include_self=False) 
            G_grid = csr_matrix.todense(G_grid)
            graph = np.multiply(G_grid, d)       # grid-connected with edge weights as latent_grid distance
            dist_matrix = dijkstra(csgraph=graph, directed=False, limit=self.limit)
            dist_matrix = torch.Tensor(dist_matrix)
            dist_matrix_lst.append(dist_matrix)

        dist = torch.stack(dist_matrix_lst).to(xA)
        dist = dist.nan_to_num(posinf=self.limit)
        
        return dist
  
class SwissRoll_geodesic(DistFunc):
    def __init__(self, **kwargs):
        self.data_cfg = kwargs.get("data_cfg", None)
        self.z_stretch = self.data_cfg.get("z_stretch", 1.0)
        pass
    
    def _xyz2th(self, xyz):
        x = xyz[:,:,0]
        y = xyz[:,:,1]
        xy = xyz[:,:,0:2]
        
        th = torch.atan2(y,x)                       # accurate up to modulo 2 * np.pi
        r = torch.sqrt((xy**2).sum(dim=2))          # r=theta for non-curvy Swiss roll
        th_true = th + 2*np.pi * torch.round((r- th)/(2* np.pi))
        return th_true
    
    def _th2s(self, th):
        s = 0.5 * (th * torch.sqrt(th**2 + 1) + torch.log(th + torch.sqrt(th**2 + 1)))
        return s
    
    def __call__(self, xA, xB, yA, yB, **kwargs):
        # hyperparameters (must change according to experimental setting)
        z_stretch = self.z_stretch
        
        catA, catB, nBatch, nA, nB, dim = self._repeat(xA, xB)
        th_catA = self._xyz2th(catA)
        th_catB = self._xyz2th(catB)
        s_catA  = self._th2s(th_catA)
        s_catB  = self._th2s(th_catB)
        z_catA = catA[:,:,2]
        z_catB = catB[:,:,2]
        dist = torch.sqrt((s_catA-s_catB)**2 + (z_stretch*(z_catA-z_catB))**2).view(nBatch, nA, nB)
        
        return dist

class TimeSteps(DistFunc):
    def __init__(self, **kwargs):
        # self.max_steps = kwargs.get("max_steps", None)
        pass
    
    def __call__(self, xA, xB, yA, yB, **kwargs):
        # ------ inputs -------
        # A,B : (Batch, T, *dims), batch of time series

        # ------ outputs -------
        # dist : (Batch, T, T), batch of dist matrices,
        # where all (T,T) matrices are same, with (i,j)-th element being |i-j|
        
        assert xA.shape==xB.shape, f'Shape mismatch in TimeSteps distfunc: got {xA.shape} and {xB.shape}.'
        
        Batch, T, *dims = xA.shape
        mat = torch.arange(start=0, end=T, step=1).repeat_interleave(T).view(T,T).to(xA)
        dist_mat = torch.abs(mat - mat.transpose(0,1))
        dist = dist_mat.repeat(Batch,1,1)
        
        # One Loop
        dist[dist>18.01] = 36.0 - dist[dist>18.01]      # shortest path on 36-loop
          
        return dist

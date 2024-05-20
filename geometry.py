#############################################
#                                           #
# code from IRVAE_public (Lee et al., 2022) #
#                                           #
#############################################

import torch
import numpy as np
from scipy import stats

from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import dijkstra

from kernels import get_distfunc

def relaxed_distortion_measure_JGinvJT(H):
    # ------ inputs -------
    # H : JGinvJT matrix for each data points, B x N x n x n (N = # of points in the graph)
    
    # ------ outputs -------
    # relaxed distortion measure, a real number
    
    TrH = H.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    TrH2 = (H @ H).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    
    return (TrH2).mean() - 2 * (TrH).mean()

def get_laplacian(X, Y, distfunc='Euclidean', bandwidth=50, **kwargs):
    # ------ inputs -------
    # X : data points in the graph, B x N x d (N = # of points in the graph, d = ambient dim.)
    # Y : targets corresponding to the data points, B x N
    # distfunc : distance function between two data points
    # bandwidth: bandwidth for exponentially decaying kernel
    
    # ------ outputs -------
    # L : Normalized Graph Laplacian, B x N x N (N = # of points in the graph)
    B, N, *dims = X.shape
    c = 1/4
    
    precomputed_dist = kwargs.get("precomputed_dist", None)
    if precomputed_dist is not None:
        # Slice the precomputed distance matrix (Batch-Kernel method)
        dist_XX = precomputed_dist
        K = torch.exp(-dist_XX**2 / bandwidth)
        d_i = K.sum(dim=1)
        D_inv = torch.diag_embed(1/d_i)
        K_tilde = D_inv @ K @ D_inv
        D_tilde_inv = torch.diag_embed(1/K_tilde.sum(dim=1))
        L = (D_tilde_inv@K_tilde - torch.diag_embed(torch.ones(B, N)).to(X))/(c*bandwidth)
    else:
        # No slicing. Compute mini-batch pairwise distance matrix (for RotatingMNIST dataset)
        dist = get_distfunc(name=distfunc, data_cfg = kwargs.get("data_cfg", None))                  
        dist_XX = dist(xA=X, xB=X, yA=Y, yB=Y)
        K = torch.exp(-dist_XX**2 / bandwidth)
        d_i = K.sum(dim=1)
        D_inv = torch.diag_embed(1/d_i)
        K_tilde = D_inv @ K @ D_inv
        D_tilde_inv = torch.diag_embed(1/K_tilde.sum(dim=1))
        L = (D_tilde_inv@K_tilde - torch.diag_embed(torch.ones(B, N)).to(X))/(c*bandwidth)
    return L
    
    
def get_JGinvJT(L, Y):
    # ------ inputs -------
    # L : Graph Laplacian, B x N x N (N = # of points in the graph)
    # Y : latent points, B x N x n (n = embedding dim.)
    
    # ------ outputs -------
    # H_tilde : JGinvJT matrix for each data points, B x N x n x n
    # H_tilde[i, j, :, :] == JGinvJT matrix for the j-th data point in the i-th data graph
    
    Batch = L.shape[0]
    N = L.shape[1]
    n = Y.shape[-1]
    
    catY1 = Y.unsqueeze(-1).repeat(1, 1, 1, n)
    catY2 = Y.unsqueeze(-2).repeat(1, 1, n, 1)
    
    term1 = catY1 * catY2
    term1 = (L @ term1.view(Batch, N, n*n)).view(Batch, N, n, n)
    
    catLY2 = (L@Y).unsqueeze(-2).repeat(1, 1, n, 1)
    term2 = catY1 * catLY2
    
    catLY1 = (L@Y).unsqueeze(-1).repeat(1, 1, 1, n)
    term3 = catY2 * catLY1
    
    H_tilde = 0.5 * (term1 - term2 - term3)
                
    return H_tilde

def get_kNNRecall(X, z, k=10, data_cfg = None):
    # ------ inputs -------
    # X: data points, (1, B, d)        d = data dim.
    # z: embedded points, (1, B, n)    n = embedding dim.
    # k: k in k-NN
    
    # ------ outputs -------
    # recall: kNN recall score (a local metric)
    
    if len(X.shape) > 3:
        X = torch.flatten(X, start_dim=2,end_dim=-1)    # (1, B, *dims) -> (1, B, d)
    
    # assert len(X.shape) == 3 and len(z.shape) == 3, f'Input shapes must have length 3, but got {len(X.shape)} and {len(z.shape)}.'
    assert X.shape[1] == z.shape[1], f'Inputs must have same shape[1], but got {X.shape[1]} and {z.shape[1]}.'
    
    zz_distfunc = 'Euclidean'
    if data_cfg is None:
        data_GT_distfunc = 'Euclidean'
    else: 
        data_GT_distfunc = data_cfg.get("data_GT_distfunc", 'Euclidean')

    # compute pairwise distance
    distfunc_XX = get_distfunc(name=data_GT_distfunc, data_cfg = data_cfg)
    distfunc_zz = get_distfunc(name=zz_distfunc)                   
    dist_XX = distfunc_XX(X,X).squeeze()     # (1, B, B) -> (B, B)
    dist_zz = distfunc_zz(z,z).squeeze()     # (1, B, B) -> (B, B)
    
    B = X.shape[1]
    dist_XX = dist_XX + torch.diag(torch.ones(B) * torch.inf).to(X)     # remove self-NN
    dist_zz = dist_zz + torch.diag(torch.ones(B) * torch.inf).to(z)     # remove self-NN
    
    # compute kNN matrix: (i,j)-th element is 1 if j is in kNN(i), 0 otherwise.
    _, indices_XX = torch.topk(dist_XX, k, dim=0, largest=False)        # indices: (k, B)
    kNN_XX = torch.zeros_like(dist_XX).to(X)                            # (B, B)
        
    for i in indices_XX:
        for j, idx in enumerate(i):
            kNN_XX[j, idx] = 1
    
    _, indices_zz= torch.topk(dist_zz, k, dim=0, largest=False)         # indices: (k, B)
    kNN_zz = torch.zeros_like(dist_zz)                                  # (B, B)
    for i in indices_zz:
        for j, idx in enumerate(i):
            kNN_zz[j, idx] = 1
    
    # evaluate which ratio of nearest neighbors in the embedding are also nearest neighbors in the data
    kNN_true_positive = kNN_XX * kNN_zz
    recall = torch.sum(kNN_true_positive) / (k*B)       # recall = (true positive) / (positive)
    
    return recall

def get_kNNRecall_Euclidean(X, z, **kwargs):
    # ------ inputs -------
    # X: data points, (1, B, d)        d = data dim.
    # z: embedded points, (1, B, n)    n = embedding dim.
    # k: k in k-NN
    
    # ------ outputs -------
    # recall: kNN recall score (a local metric)
    
    if len(X.shape) > 3:
        X = torch.flatten(X, start_dim=2,end_dim=-1)    # (1, B, *dims) -> (1, B, d)
    
    B = X.shape[1]
    recall_lst = []
    
    X_ = X.squeeze().detach().cpu().numpy()
    z_ = z.squeeze().detach().cpu().numpy()
    
    for k in np.int_(np.linspace(start=10, stop=200, num=20)):
        # Average over a range of k values from 10 to 200 in stpes of 10, as proposed in "Topological Autoencoders," Moor et al. (2020).
        G_X = kneighbors_graph(X_, n_neighbors=k, mode="connectivity", include_self=False)
        kNN_XX = csr_matrix(G_X).toarray()
        G_z = kneighbors_graph(z_, n_neighbors=k, mode="connectivity", include_self=False)
        kNN_zz = csr_matrix(G_z).toarray()
        
        kNN_true_positive = kNN_XX * kNN_zz
        recall = np.sum(kNN_true_positive) / (k*B)
        recall_lst.append(recall)
        
        if k > (X.shape[0] - 1):
            break
    
    return torch.tensor(recall_lst).mean()

def get_SpearmanCorrelation(x, y, z, data_cfg = None):
    # ------ inputs -------
    # x: data points, (1, B, d) or (1, B, *dims)    d = data dim.
    # y: targets, (1, B) or (1, B, ydim)            ydim = target dim. (5 for dSprites)
    # z: latent points, (1, B, n)                 n = embedding dim.
    
    # ------ outputs -------
    # corr: Spearman correlation of pairwise distances (rank-based correlation)
    
    if len(x.shape) > 3:
        x = torch.flatten(x, start_dim=2,end_dim=-1)    # (1, B, *dims) -> (1, B, d)
    
    if data_cfg is None:
        data_GT_distfunc = 'Euclidean'
    else: 
        data_GT_distfunc = data_cfg.get("data_GT_distfunc", 'Euclidean')
    
    # compute pairwise distance
    distfunc_xx = get_distfunc(name=data_GT_distfunc, data_cfg = data_cfg) 
    distfunc_zz = get_distfunc(name='Euclidean')                   
    dist_xx = distfunc_xx(xA=x,xB=x,yA=y,yB=y).squeeze()        # (1, B, B) -> (B, B)
    dist_zz = distfunc_zz(z,z).squeeze()                        # (1, B, B) -> (B, B)
    
    # convert pairwise distance into 1D tensor
    B = x.shape[1]
    indices = torch.triu_indices(B, B, offset=1)

    dist_xx_vec = dist_xx[indices[0], indices[1]]
    dist_zz_vec = dist_zz[indices[0], indices[1]]
    
    # compute Spearman correlation
    corr, _ = stats.spearmanr(dist_xx_vec.detach().cpu().numpy(), dist_zz_vec.detach().cpu().numpy())
    corr = torch.Tensor([corr]).to(x)
    
    return corr

def get_density_kl(X, z, sigma=0.1, data_cfg = None):    
    # ------ inputs -------
    # X: data points, (1, B, d)        d = data dim.
    # z: embedded points, (1, B, n)    n = embedding dim.
    
    # ------ outputs -------
    # kl: KL divergence between density esimator (Chazal et al., 2011)
    if len(X.shape) > 3:
        X = torch.flatten(X, start_dim=2,end_dim=-1)    
    
    zz_distfunc = 'Euclidean'
    if data_cfg is None:
        data_GT_distfunc = 'Euclidean'
    else: 
        data_GT_distfunc = data_cfg.get("data_GT_distfunc", 'Euclidean')
    
    # compute pairwise distance
    distfunc_XX = get_distfunc(name=data_GT_distfunc, data_cfg = data_cfg) 
    distfunc_zz = get_distfunc(name=zz_distfunc)                   
    dist_XX = distfunc_XX(X,X).squeeze()     # (1, B, B) -> (B, B)
    dist_zz = distfunc_zz(z,z).squeeze()     # (1, B, B) -> (B, B)
    
    # compute density estimator
    dist_XX_normalized = dist_XX / torch.max(dist_XX)
    density_X = torch.sum(torch.exp(-(dist_XX_normalized ** 2) / sigma), dim=-1)
    density_X_normalized = density_X / torch.sum(density_X)
    dist_zz_normalized = dist_zz / torch.max(dist_zz)
    density_z = torch.sum(torch.exp(-(dist_zz_normalized ** 2) / sigma), dim=-1)
    density_z_normalized = density_z / torch.sum(density_z)
    
    # compute density KL divergence
    kl = torch.sum(density_X_normalized *(torch.log(density_X_normalized) - torch.log(density_z_normalized)))
    
    return kl
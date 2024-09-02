#############################################
#                                           #
# code from IRVAE_public (Lee et al., 2022) #
#                                           #
#############################################

from torch.utils import data

from loader.MNIST_dataset import MNIST, RotatingMNIST
from loader.Synthetic_dataset import SwissRoll
from loader.dSprites_dataset import dSprites

from loader.collate_fn import *

def get_dataloader(data_dict, loader_device='cuda:0', **kwargs):
    dataset = get_dataset(data_dict, **kwargs)
    verbose = kwargs.get('verbose', False)
        
    if hasattr(dataset, 'precompute_global_dist'):
        precompute_global_dist=dataset.precompute_global_dist
    else:
        precompute_global_dist=False
        
    if hasattr(dataset, 'global_dist'):
        global_dist=dataset.global_dist
    else:
        global_dist=None

    if verbose:
        print("Collate_fn:", data_dict.get("collate_fn", None))
    collate_fn = get_collate_fn(
        data_dict, 
        loader_device,
        precompute_global_dist=precompute_global_dist, 
        global_dist=global_dist,
        **kwargs)
        
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        shuffle=data_dict.get("shuffle", True),
        collate_fn=collate_fn,
    )
    return loader

def get_dataset(data_dict, **kwargs):
    name = data_dict["dataset"]
    if name == 'MNIST':
        dataset = MNIST(**data_dict, **kwargs)
    elif name == 'RotatingMNIST':
        dataset = RotatingMNIST(**data_dict, **kwargs)
    elif name == 'SwissRoll':
        dataset = SwissRoll(**data_dict, **kwargs)
    elif name == 'dSprites':
        dataset = dSprites(**data_dict, **kwargs)
    else:
        raise NotImplementedError
    
    return dataset

def get_collate_fn(data_dict, device, **kwargs):
    name = data_dict.get("collate_fn", None)
    if name == 'laplacian_collate_fn':                                                  # for ERAE 
        collate_fn = Laplacian_collate_fn(data_dict, device, **kwargs)
    elif name == 'timeseries_collate_fn':                                               # for TS_AE
        collate_fn = TimeSeries_collate_fn(data_dict, device, **kwargs)
    elif name == 'timeseries_laplacian_collate_fn':                                     # for TS_ERAE
        collate_fn = TimeSeries_Laplacian_collate_fn(data_dict, device, **kwargs)
    else:
        collate_fn = None
    
    return collate_fn
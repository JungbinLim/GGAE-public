#############################################
#                                           #
# code from IRVAE_public (Lee et al., 2022) #
#                                           #
#############################################

import os
import time
import math

import numpy as np
import torch
from metrics import averageMeter

class BaseTrainer:
    """Trainer for AE"""
    def __init__(self, optimizer, training_cfg, device):
        self.training_cfg = training_cfg
        self.device = device
        self.optimizer = optimizer

    def train(self, model, d_dataloaders, logger=None, logdir=""):
        cfg = self.training_cfg
    
        time_meter = averageMeter()
        train_loader, val_loader = (d_dataloaders["training"], d_dataloaders["validation"])
        kwargs = {'dataset_size': len(train_loader.dataset)}
        i_iter = 0
        best_val_loss = np.inf
    
        for i_epoch in range(1, cfg['n_epoch'] + 1):
            for x, _ in train_loader:
                i_iter += 1

                model.train()
                start_ts = time.time()
                d_train = model.train_step(x.to(self.device), optimizer=self.optimizer, **kwargs)
                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)

                if i_iter % cfg.print_interval == 0:
                    d_train = logger.summary_train(i_iter)
                    print(
                        f"Epoch [{i_epoch:d}] \nIter [{i_iter:d}]\tAvg Loss: {d_train['loss/train_loss_']:.6f}\tElapsed time: {time_meter.sum:.4f}"
                    )
                    time_meter.reset()

                model.eval()
                if i_iter % cfg.val_interval == 0:
                    for x, _ in val_loader:
                        d_val = model.validation_step(x.to(self.device))
                        logger.process_iter_val(d_val)
                    d_val = logger.summary_val(i_iter)
                    val_loss = d_val['loss/val_loss_']
                    print(d_val['print_str'])
                    best_model = val_loss < best_val_loss

                    if best_model:
                        print(f'Iter [{i_iter:d}] best model saved {val_loss:.6f} <= {best_val_loss:.6f}')
                        best_val_loss = val_loss
                        self.save_model(model, logdir, best=True)
                    
                    d_eval = model.eval_step(val_loader, device=self.device)
                    logger.add_val(i_iter, d_eval)
                    print_str = f'Iter [{i_iter:d}]'
                    for key, val in d_eval.items():
                        if key.endswith('_'):
                            print_str = print_str + f'\t{key[:-1]}: {val:.4f}'
                    print(print_str)

                if i_iter % cfg.visualize_interval == 0:
                    d_val = model.visualization_step(train_loader, device=self.device)
                    logger.add_val(i_iter, d_val)

        self.save_model(model, logdir, i_iter="last")
        return model, best_val_loss

    def save_model(self, model, logdir, best=False, i_iter=None, i_epoch=None, best_eval=False):
        if best:
            pkl_name = "model_best.pkl"
        elif best_eval:
            pkl_name = "model_eval_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = f"model_iter_{i_iter}.pkl"
            else:
                pkl_name = f"model_epoch_{i_epoch}.pkl"
        state = {"epoch": i_epoch, "iter": i_iter, "model_state": model.state_dict()}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f"Model saved: {pkl_name}")

class GraphTrainer:
    """Trainer to use collate_fn"""
    def __init__(self, optimizer, training_cfg, data_cfg, device, **kwargs):
        self.training_cfg = training_cfg
        self.data_cfg = data_cfg
        self.device = device
        self.optimizer = optimizer

    def train(self, model, d_dataloaders, logger=None, logdir=""):
        cfg = self.training_cfg
    
        time_meter = averageMeter()
        train_loader, val_loader = (d_dataloaders["training"], d_dataloaders["validation"])

        i_iter = 0
        best_val_loss = np.inf
        save_best_eval = cfg.get("save_best_eval", None)
        if save_best_eval is not None:
            best_eval_loss = np.inf
        eval_interval = cfg.get("eval_interval", cfg.val_interval)
            
        for i_epoch in range(1, cfg['n_epoch'] + 1):
            for x, _, collated_obj in train_loader:
                i_iter += 1

                model.train()
                start_ts = time.time()
                d_train = model.train_step(
                    x.to(self.device), 
                    collated_obj=collated_obj.to(self.device), 
                    optimizer=self.optimizer, 
                    dataset_size=len(train_loader.dataset),
                    data_cfg = self.data_cfg,
                    bandwidth = self.data_cfg.get("bandwidth", 0.1),
                    distfunc = self.data_cfg.get("distfunc", "Euclidean")
                )
                
                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)
                
                if i_iter % cfg.print_interval == 0:
                    d_train = logger.summary_train(i_iter)
                    print(
                        f"Epoch [{i_epoch:d}] \nIter [{i_iter:d}]\tAvg Loss: {d_train['loss/train_loss_']:.6f}\tElapsed time: {time_meter.sum:.4f}"
                    )
                    time_meter.reset()

                model.eval()
                if i_iter % cfg.val_interval == 0:
                    for x, _, collated_obj in val_loader: 
                        d_val = model.validation_step(
                            x.to(self.device), 
                            collated_obj=collated_obj.to(self.device),
                            dataset_size=len(val_loader.dataset),
                            data_cfg = self.data_cfg,
                        )
                        logger.process_iter_val(d_val)
                        
                    d_val = logger.summary_val(i_iter)
                    val_loss = d_val['loss/val_loss_']
                    print(d_val['print_str'])
                    best_model = val_loss < best_val_loss

                    if best_model:
                        print(f'Iter [{i_iter:d}] best model saved {val_loss:.6f} <= {best_val_loss:.6f}')
                        best_val_loss = val_loss
                        self.save_model(model, logdir, best=True)
                
                if i_iter % eval_interval == 0:
                    d_eval = model.eval_step(
                        val_loader, 
                        device=self.device, 
                        data_cfg = self.data_cfg,
                        dataset_size=len(val_loader.dataset),
                    )
                    
                    logger.add_val(i_iter, d_eval)
                    print_str = f'Iter [{i_iter:d}]'
                    for key, val in d_eval.items():
                        if key.endswith('_'):
                            print_str = print_str + f'\t{key[:-1]}: {val:.4f}'
                    print(print_str)
                    
                    if save_best_eval is not None:
                        eval_loss = d_eval[f'eval/{save_best_eval}']
                        if eval_loss < best_eval_loss:
                            print(f'Iter [{i_iter:d}] best eval model saved {eval_loss:.6f} <= {best_eval_loss:.6f}')
                            best_eval_loss = eval_loss
                            self.save_model(model, logdir, best_eval=True, i_iter=i_iter)
                        
                if i_iter % cfg.visualize_interval == 0:
                    d_val = model.visualization_step(
                        train_loader, 
                        device=self.device,
                        dataset_size=len(val_loader.dataset),
                    )
                    logger.add_val(i_iter, d_val)

        self.save_model(model, logdir, i_iter="last")
        return model, best_val_loss
        
    def save_model(self, model, logdir, best=False, i_iter=None, i_epoch=None, best_eval=False):
        if best:
            pkl_name = "model_best.pkl"
        elif best_eval:
            pkl_name = "model_eval_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = f"model_iter_{i_iter}.pkl"
            else:
                pkl_name = f"model_epoch_{i_epoch}.pkl"
        state = {"epoch": i_epoch, "iter": i_iter, "model_state": model.state_dict()}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f"Model saved: {pkl_name}")
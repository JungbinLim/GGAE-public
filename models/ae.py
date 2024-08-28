#############################################
#                                           #
# code from IRVAE_public (Lee et al., 2022) #
#                                           #
#############################################

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from torchvision.utils import make_grid
from utils.utils import label_to_color, figure_to_array
from geometry import get_JGinvJT, relaxed_distortion_measure_JGinvJT, get_SpearmanCorrelation


def visualize_MNIST(model, dl, device, **kwargs):
    results = {}
    
    num_figures = 100
    num_each_axis = 10
    B, *dims = dl.dataset.data.shape
    x = dl.dataset.data[torch.randperm(B)[:num_figures]]
        
    recon = model.decode(model.encode(x.to(device)))
    x_img = make_grid(x.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
    recon_img = make_grid(recon.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
    
    C, H, W = x_img.shape
    results['vis/input_and_recon@'] = torch.cat([torch.clip(x_img, min=0, max=1), torch.ones(C, H, 2)*0.5, torch.clip(recon_img, min=0, max=1)], dim=2)
    
    is_z2 = (model.encode(x.to(device)).shape[-1] == 2)
    if is_z2:
        num_latent_points = 500
        rand_idx = torch.randperm(B)[:num_latent_points]
        x = dl.dataset.data[rand_idx]
        targets = dl.dataset.targets[rand_idx].squeeze().detach().cpu()
        z = model.encode(x.to(device)).detach().cpu()
        colors = label_to_color(targets.numpy())
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        plot_z = z.detach().cpu()
        for label in np.unique(targets.numpy()):
            label_idx = targets == label
            axs[0].scatter(plot_z[label_idx, 0], plot_z[label_idx, 1], marker='o', c=colors[label_idx]/255.0, s=5, label=label)
        axs[0].set_title('Pullback')
        axs[0].axis('equal')
            
        for label in np.unique(targets.numpy()):
            label_idx = targets == label
            axs[1].scatter(plot_z[label_idx, 0], plot_z[label_idx, 1], marker='o', c=colors[label_idx]/255.0, s=5, label=label)
            
        axs[1].set_title('Pushforward')
        axs[1].axis('equal')
        
        plt.legend()
        plt.close()
        img_latent = np.transpose(figure_to_array(fig), (2, 0, 1))[:3,:,:]
        results['vis/latent_space@'] = img_latent
    
    return results

def visualize_SwissRoll(model, dl, device, **kwargs):
    results = {}
    
    x = dl.dataset.data                                 # (ttl, xdim)
    targets = dl.dataset.targets                        # (ttl)
    z = model.encode(x.to(device)).detach().cpu()       # (ttl, zdim)
    recon = model.decode(z.to(device)).detach().cpu()   # (ttl, xdim)
    
    plot_input = x.detach().cpu().clone()
    plot_recon = recon.clone()
    plot_target = targets.clone()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=plot_input[:, 0], ys=plot_input[:, 1], zs=plot_input[:, 2], marker='o', c=plot_target)
    ax.set_title('Input')
    plt.close()
    img_input = np.transpose(figure_to_array(fig), (2, 0, 1))[:3,:,:]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=plot_recon[:, 0], ys=plot_recon[:, 1], zs=plot_recon[:, 2], marker='o', c=plot_target)
    ax.set_title('Recon')
    plt.close()
    img_recon = np.transpose(figure_to_array(fig), (2, 0, 1))[:3,:,:]
    
    img_input_and_recon = np.concatenate([img_input, img_recon], axis=2)
    results['vis/input_and_recon@'] = img_input_and_recon
    
    is_z2 = (z.shape[-1] == 2)
    if is_z2:
        plot_z = z.clone()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.scatter(plot_z[:, 0], plot_z[:, 1], marker='o', c=targets, s=1)
        ax.set_title('2D Latent Space')
        ax.axis('equal')
        plt.close()
        img_latent = np.transpose(figure_to_array(fig), (2, 0, 1))[:3,:,:]
        results['vis/latent_space@'] = img_latent
    return results

def visualize_dSprites(model, dl, device, **kwargs):
    results = {}
    
    x = dl.dataset.data                                 # (ttl, *dims)
    targets = dl.dataset.targets                        # (ttl, ydim=5)
    z = model.encode(x.to(device)).detach().cpu()       # (ttl, zdim)
    recon = model.decode(z.to(device)).detach().cpu()   # (ttl, *dims)
    
    num_figures = 25
    num_each_axis = 5
    B, *dims = x.shape
    x_input_recon = x[torch.randperm(B)[:num_figures]]
        
    recon = model.decode(model.encode(x_input_recon.to(device)))
    x_img = make_grid(x_input_recon.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
    recon_img = make_grid(recon.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)

    C, H, W = x_img.shape
    results['vis/input_and_recon@'] = torch.cat([torch.clip(x_img, min=0, max=1), torch.ones(C, H, 2)*0.5, torch.clip(recon_img, min=0, max=1)], dim=2)
    
    is_z3 = (z.shape[-1] == 3)
    if is_z3:
        plot_z = z.clone()
        color = targets[:, 1].detach().cpu().numpy()    # color based on scale
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(plot_z[:, 0], plot_z[:, 1], plot_z[:, 2], marker='o', c=color, s=9)
        ax.set_title('3D Latent Space')
        plt.close()
        img_latent = np.transpose(figure_to_array(fig), (2, 0, 1))[:3,:,:]
        results['vis/latent_space@'] = img_latent
    
    return results

class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
    
    def validation_step(self, x, **kwargs):
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        return {"loss": loss.item()}
    
    def eval_step(self, dl, device, **kwargs):
        '''
        dl : dataloader
        device : torch cuda device
        kwargs : may include data config
        '''
        x = dl.dataset.data.to(device)          # (N,*dims) or (N,d)
        y = dl.dataset.targets.to(device)       # (N) or (N,ydim)
        z = self.encode(x)                      # (N,n)
        recon = self.decode(z)                  # (N,*dims) or (N,d)
        recon_loss = ((recon - x)**2).mean().item()
        
        data_cfg = kwargs.get("data_cfg", None)
        Spearman_correlation = get_SpearmanCorrelation(x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0), data_cfg = data_cfg).item()     # based on GT data dist
        
        return {
            'eval/recon_loss_': recon_loss,
            'eval/Spearmen_correlation_GT_': Spearman_correlation,
        }
        
    def visualization_step(self, dl, device, **kwargs):
        name = dl.dataset.__class__.__name__
        if 'MNIST' in name:
            results = visualize_MNIST(self, dl, device, **kwargs)
        elif ('SwissRoll' in name) or ('Cylinder' in name) or ('S_manifold' in name) or ('Hemisphere' in name) or ('Sphere' in name) or ('MobiusStrip' in name) or ('Helix' in name):
            results = visualize_SwissRoll(self, dl, device, **kwargs)
        elif 'dSprites' in name:
            results = visualize_dSprites(self, dl, device, **kwargs)
        else:
            results = {}
        return results

class GGAE(AE):
    def __init__(self, encoder, decoder, iso_reg=1.0):
        super(GGAE, self).__init__(encoder, decoder)
        self.iso_reg = iso_reg
        
    def train_step(self, x, collated_obj, optimizer, **kwargs):
        '''
        Using Laplacian_collate_fn,
        
        x : (Batch, *data_dims), A batch of data points 
        collated_obj(L) : (1, Batch, Batch), Laplacian matrix of the batch
        '''
        optimizer.zero_grad()
        
        L = collated_obj
        z = self.encode(x)
        H_tilde = get_JGinvJT(L, z.unsqueeze(0))
        iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde)
        
        recon = self(x)
        recon_loss = ((recon - x)**2).mean()
        loss = recon_loss + self.iso_reg * iso_loss
        
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'loss/train_recon_loss_': recon_loss.item(),
            'loss/train_iso_loss_': iso_loss.item(),
        }
    
    def validation_step(self, x, collated_obj, **kwargs):
        
        L = collated_obj
        z = self.encode(x)
        H_tilde = get_JGinvJT(L, z.unsqueeze(0))
        iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde)
        
        recon = self(x)
        recon_loss = ((recon - x)**2).mean()
        loss = recon_loss + self.iso_reg * iso_loss
        
        return {
            'loss': loss.item(),
            'loss/valid_recon_loss_': recon_loss.item(),
            'loss/valid_iso_loss_': iso_loss.item(),
        }


class TIMESERIES_AE(AE):
    def __init__(self, encoder, decoder):
        super(TIMESERIES_AE, self).__init__(encoder, decoder)
        
    def encode(self, x):
        B, T, *dims = x.shape
        x = x.flatten(0,1)
        z = self.encoder(x)
        z = z.view(B,T,-1)
        return z

    def decode(self, z):
        B, T, n = z.shape
        z = z.flatten(0,1)
        x = self.decoder(z)
        x = x.unflatten(0, (B,T))
        return x
    
    def forward(self, x):
        z = self.encode(x)      # (B,T,*dims) -> (B*T, n)
        recon = self.decode(z)  # (B*T, n) -> (B*T, *dims)
        return recon
        
    def train_step(self, x, optimizer, **kwargs):
        '''
        Using timeseries_collate_fn,
        
            x: (B, T, *dims)
        '''
        
        print("Timeseries AE train_step, x.shape", x.shape)
        
        optimizer.zero_grad()
        
        recon = self(x)
        loss = ((recon - x)**2).mean()
        
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
    
    def validation_step(self, x, **kwargs):
        recon = self(x)
        loss = ((recon - x)**2).mean()
        return {"loss": loss.item()}
    
    def eval_step(self, dl, device, **kwargs):
        '''
        dl : dataloader
        device : torch cuda device
        kwargs : may include data config
        '''
        x = dl.dataset.data.to(device)                  # (ttl, *dims, T)
        x = x.unsqueeze(1).transpose(1,-1).squeeze(-1)  # (ttl, T, *dims)
        y = dl.dataset.targets.to(device)
        z = self.encode(x)
        recon = self.decode(z)
        recon_loss = ((recon - x)**2).mean().item()
        
        data_cfg = kwargs.get("data_cfg", None)
        # Compute Spearman correlation for each time series
        spear = []
        for i in range(100):
            xx = x[i].unsqueeze(0)
            yy = y[i].unsqueeze(0)
            zz = z[i].unsqueeze(0)
            spear.append(get_SpearmanCorrelation(xx, yy, zz, data_cfg = data_cfg).item())
        Spearman_correlation = np.mean(spear)
        
        return {
            'eval/recon_loss_': recon_loss,
            'eval/Spearmen_correlation_GT_': Spearman_correlation,
        }
    
    def visualization_step(self, dl, device, **kwargs):
        '''
        dl : dataloader
        device : torch cuda device
        '''
        
        if 'MNIST' in dl.dataset.__class__.__name__:
            results = self.visualize_MNIST(dl, device, **kwargs)
        else:
            results = {}

        return results
    
    def visualize_MNIST(self, dl, device, **kwargs):
        '''
        dl.dataset.data.shape    = (ttl, *dims, T)
        dl.dataset.targets.shape = (ttl)
        '''
        
        results = {}
        
        num_each_axis = dl.dataset.data.shape[-1]   
        num_figures = 10                            
        
        T = dl.dataset.data.shape[-1]
        
        data = dl.dataset.data.unsqueeze(1).transpose(1,-1).squeeze(-1)
        targets = torch.repeat_interleave(dl.dataset.targets, T).view(-1,T)
        
        B, *dims = data.shape
        x = data[torch.randperm(B)[:num_figures]]       # (num_fig, T, *dims)
        recon = self.decode(self.encode(x.to(device)))  # (num_fig, T, *dims)
        
        x_flattened = x.flatten(0,1)                    # (num_fig*T, *dims)
        recon_flattened = recon.flatten(0,1)            # (num_fig*T, *dims)
        x_img = make_grid(x_flattened.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        recon_img = make_grid(recon_flattened.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        
        C, H, W = x_img.shape
        results['vis/input_and_recon@'] = torch.cat([torch.clip(x_img, min=0, max=1), torch.ones(C, H, 2)*0.5, torch.clip(recon_img, min=0, max=1)], dim=2)
        
        is_z2 = (self.encode(x.to(device)).shape[-1] == 2)
        if is_z2:
            
            num_latent_points = 100
            rand_idx = torch.randperm(B)[:num_latent_points]
            x = data[rand_idx]      # (num_latent_points, T, *dims)
            
            targets = targets[rand_idx].flatten(0,1).squeeze().detach().cpu()
            
            z = self.encode(x.to(device)).detach().cpu().flatten(0,1)   # (B*T, 2)
            colors = label_to_color(targets.numpy())
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot()
                
            z_rollout = []
            o_rollout = []

            plot_z = z.detach().cpu()
            for label in np.unique(targets.numpy()):
                label_idx = targets == label
                
                z_label = z[label_idx, :]
                lstates = z_label[:T].to(device)
                o_recon = self.decoder(lstates)
                z_rollout.append(lstates.unsqueeze(0))
                o_rollout.append(o_recon.unsqueeze(0))
                
                ax.scatter(plot_z[label_idx, 0], plot_z[label_idx, 1], marker='o', c=colors[label_idx]/255.0, s=5, label=label)

            z_rollout = torch.cat(z_rollout, dim=0).detach().cpu().numpy()  # (# of labels, T, 2)
            o_rollout = torch.cat(o_rollout, dim=0).detach().cpu().numpy()  # (# of labels, T, *dims)
            
            # plot a time series using AnnotationBbox on axs[0]               
            for idx in range(len(z_rollout)):
                plt.plot(z_rollout[idx, :, 0], z_rollout[idx, :, 1], '-o', color='k')
                for t_idx in range(z_rollout.shape[1]):
                    im = OffsetImage(o_rollout[idx, t_idx, :, :, :].transpose(1, 2, 0), zoom=0.7)
                    if t_idx < T:             # given observations
                        ab = AnnotationBbox(im, (z_rollout[idx, t_idx, 0], z_rollout[idx, t_idx, 1]), xycoords='data', frameon=True)
                    elif t_idx == z_rollout.shape[1]-1:     # the final frame of rollout
                        ab = AnnotationBbox(im, (z_rollout[idx, t_idx, 0], z_rollout[idx, t_idx, 1]), xycoords='data', frameon=True, bboxprops=dict(edgecolor='red'))    
                    else:
                        ab = AnnotationBbox(im, (z_rollout[idx, t_idx, 0], z_rollout[idx, t_idx, 1]), xycoords='data', frameon=False)
                    plt.gca().add_artist(ab)
            
            ax.set_title('2D Latent Space')
            ax.axis('equal')
            plt.legend()
            plt.close()
            img_latent = np.transpose(figure_to_array(fig), (2, 0, 1))[:3,:,:]
            results['vis/latent_space@'] = img_latent
        
        return results

class TIMESERIES_GGAE(TIMESERIES_AE):
    def __init__(self, encoder, decoder, iso_reg=1.0):
        super(TIMESERIES_GGAE, self).__init__(encoder, decoder)
        self.iso_reg = iso_reg
        
    def train_step(self, x, collated_obj, optimizer, **kwargs):
        '''
        Using timeseries_laplacian_collate_fn,
        
            x: (B, T, *dims)
            collated_obj: (B, T, T)  series-wise Laplacians
        '''
        
        optimizer.zero_grad()
        L = collated_obj
        z = self.encode(x)
        H_tilde = get_JGinvJT(L, z)
        iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde)
        
        recon = self(x)
        recon_loss = ((recon - x)**2).mean()
        loss = recon_loss + self.iso_reg * iso_loss
        
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'loss/train_recon_loss_': recon_loss.item(),
            'loss/train_iso_loss_': iso_loss.item(),
        }
    
    def validation_step(self, x, collated_obj, **kwargs):
        
        L = collated_obj
        z = self.encode(x)
        H_tilde = get_JGinvJT(L, z)
        iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde)
        
        recon = self(x)
        recon_loss = ((recon - x)**2).mean()
        loss = recon_loss + self.iso_reg * iso_loss
        
        return {
            'loss': loss.item(),
            'loss/valid_recon_loss_': recon_loss.item(),
            'loss/valid_iso_loss_': iso_loss.item(),
        }
#############################################
#                                           #
# code from dreamer-pytorch (Planet, 2018)  #
#                                           #
#############################################

import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torchvision.transforms.functional import affine

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from geometry import (
    # get_pullbacked_Riemannian_metric,
    # relaxed_distortion_measure,
    # get_flattening_scores,
    get_JGinvJT,
    relaxed_distortion_measure_JGinvJT,
    # boundary_loss,
    get_laplacian,
)
from utils.utils import label_to_color, PD_metric_to_ellipse, figure_to_array, plotly_fig2array

import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

class VisualEncoder(nn.Module):
    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        if embedding_size == 1024:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(1024, embedding_size)

    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)
        return hidden

class VisualDecoder_hstateconnected(nn.Module):
    def __init__(self,
            hstate_size,
            lstate_size,
            embedding_size,
            activation_function='relu'
        ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(lstate_size + hstate_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, state, latent):
        hidden = self.fc1(torch.cat([state, latent], dim=1))
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation

class VisualDecoder(nn.Module):
    def __init__(self,
            lstate_size,
            embedding_size,
            activation_function='relu'
        ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(lstate_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, lstate):
        hidden = self.fc1(lstate)
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation

class RecurrentStateSpaceModel(nn.Module):
    """Recurrent State Space Model for no-action, no-reward, only-dynamics setting, no hstate connection
    """
    def __init__(self,
            encoder,
            decoder,
            hstate_dim=200,
            hidden_dim=200,
            activation='relu'
        ):
        super().__init__()
        self.hstate_dim = hstate_dim
        self.lstate_size = int(encoder.out_chan / 2)
        self.act_fn = getattr(F, activation)
        self.encoder = encoder
        self.decoder = decoder
        
        self.grucell = nn.GRUCell(self.hstate_dim, self.hstate_dim)
        self.lat_act_layer = nn.Linear(self.lstate_size, self.hstate_dim)
        
        self.fc_prior_1 = nn.Linear(self.hstate_dim, hidden_dim)
        self.fc_prior_m = nn.Linear(hidden_dim, self.lstate_size)
        self.fc_prior_s = nn.Linear(hidden_dim, self.lstate_size)

    def get_init_state(self, enc, h_t=None, s_t=None, sample=False):
        """Returns the initial posterior given the observation."""
        N, dev = enc.size(0), enc.device
        h_t = torch.zeros(N, self.hstate_dim).to(dev) if h_t is None else h_t
        s_t = torch.zeros(N, self.lstate_size).to(dev) if s_t is None else s_t
        h_tp1 = self.deterministic_state_fwd(h_t, s_t)
        if sample:
            s_tp1 = self.state_posterior(enc, sample=True)
        else:
            s_tp1, _ = self.state_posterior(enc)
        return h_tp1, s_tp1

    def deterministic_state_fwd(self, h_t, s_t):
        """Returns the deterministic state given the previous states
        and action.
        """
        h = self.act_fn(self.lat_act_layer(s_t))
        return self.grucell(h, h_t)

    def state_prior(self, h_t, sample=False):
        """Returns the state prior given the deterministic state."""
        z = self.act_fn(self.fc_prior_1(h_t))
        m = self.fc_prior_m(z)
        s = F.softplus(self.fc_prior_s(z)) + 1e-1
        if sample:
            return m + torch.randn_like(m) * s
        return m, s

    def state_posterior(self, o_t, sample=False):
        """Returns the state prior given the deterministic state and obs."""
        z = self.encoder(o_t)
        m = z[:, :self.lstate_size]
        s = F.softplus(z[:, self.lstate_size:]) + 1e-1
        if sample:
            return m + torch.randn_like(m) * s
        return m, s

    def rollout_prior(self, o_t, T, h_t=None, s_t=None):
        h_t, s_t = self.get_init_state(o_t, h_t=h_t, s_t=s_t, sample=False)
        o_t_recon = self.decoder(s_t)
        
        lstates, o_recon = [s_t], [o_t_recon]
        
        for t_idx in range(1, T):
            h_tp1 = self.deterministic_state_fwd(h_t, s_t)
            s_tp1_pr_m, s_tp1_pr_s = self.state_prior(h_tp1)
            s_tp1_prior_dist = Normal(s_tp1_pr_m, s_tp1_pr_s)
            
            s_tp1 = s_tp1_prior_dist.rsample()
            o_tp1_recon = self.decoder(s_tp1)
            
            lstates.append(s_tp1)
            o_recon.append(o_tp1_recon)
            
            h_t, s_t = h_tp1, s_tp1
            
        return torch.stack(lstates, dim=-1), torch.stack(o_recon, dim=-1)
    
    def rollout_multiple_observation(self, o, T_rollout, h_t=None, s_t=None):
        """
        rollout with multiple frames of observation. rollout_prior can be replaced by this function,
        since single frame o_t will automatically skip the forward process because T_obs = 1.
        """ 
        
        ### forward on posteriors using observation (but no prior and recon are needed) ###
        """code from the forward function"""
        
        ## get the number of frames of observation
        if len(o.shape)==4:
            # (B, C, W, H)..single frame
            o = o.unsqueeze(-1)     # make it into (B, C, W, H, 1)
        elif len(o.shape)==5:
            # (B, C, W, H, T_obs).. multiple frames
            pass
        T_obs = o.shape[-1]         # number of frames of observation (becomes 1 if single frame, n if n frames)
        
        # o_recon    = []
        # priors     = []
        posteriors = []
        
        o_t = o[..., 0]     # first observation
        
        h_t, s_t = self.get_init_state(o_t, sample=False)       # h_1, s_1

        # o_t_recon = self.decoder(s_t)
        # o_recon.append(o_t_recon)

        for t_idx in range(1, T_obs):

            h_tp1 = self.deterministic_state_fwd(h_t, s_t)
            # s_tp1_pr_m, s_tp1_pr_s = self.state_prior(h_tp1)
            # s_tp1_prior_dist = Normal(s_tp1_pr_m, s_tp1_pr_s)

            o_tp1 = o[..., t_idx]
            s_tp1_po_m, s_tp1_po_s = self.state_posterior(o_tp1)
            s_tp1_posterior_dist = Normal(s_tp1_po_m, s_tp1_po_s)

            # s_tp1 = self.state_posterior(o_tp1, sample=True)
            # s_tp1 = s_tp1_prior_dist.rsample()
            s_tp1 = s_tp1_posterior_dist.rsample()
            # o_tp1_recon = self.decoder(s_tp1)

            # o_recon.append(o_tp1_recon)
            # priors.append([s_tp1_pr_m, s_tp1_pr_s])
            posteriors.append([s_tp1_po_m, s_tp1_po_s])

            h_t, s_t = h_tp1, s_tp1                             # In the last iteration, h_{T_obs}, s_{T_obs}
            
            
        ### rollout using the output (h_t, s_t) of forward ###
        """code from the rollout_prior function"""
        # h_t, s_t = self.get_init_state(o_t, h_t=h_t, s_t=s_t, sample=False)         # h_{T_obs + 1}, s_{T_obs + 1}
        # o_t_recon = self.decoder(s_t)
        
        # lstates, o_recon = [s_t], [o_t_recon]
        lstates, o_recon = [], []
        
        
        # rollout starting from the final (h_t, s_t) obtained in forward process.
        for t_idx in range(T_rollout):
            h_tp1 = self.deterministic_state_fwd(h_t, s_t)
            s_tp1_pr_m, s_tp1_pr_s = self.state_prior(h_tp1)
            s_tp1_prior_dist = Normal(s_tp1_pr_m, s_tp1_pr_s)
            
            s_tp1 = s_tp1_prior_dist.rsample()
            o_tp1_recon = self.decoder(s_tp1)
            
            lstates.append(s_tp1)
            o_recon.append(o_tp1_recon)
            
            h_t, s_t = h_tp1, s_tp1
            
        return torch.stack(lstates, dim=-1), torch.stack(o_recon, dim=-1)

    def get_device(self):
        return list(self.parameters())[0].device
    
    def forward(self, o):
        B, C, W, H, T = o.shape
        
        o_recon, priors, posteriors = [], [], []
        
        # t=0
        o_t = o[..., 0]
        h_t, s_t = self.get_init_state(o_t, sample=False)

        o_t_recon = self.decoder(s_t)
        o_recon.append(o_t_recon)

        for t_idx in range(1, T):

            h_tp1 = self.deterministic_state_fwd(h_t, s_t)
            s_tp1_pr_m, s_tp1_pr_s = self.state_prior(h_tp1)
            s_tp1_prior_dist = Normal(s_tp1_pr_m, s_tp1_pr_s)

            o_tp1 = o[..., t_idx]
            s_tp1_po_m, s_tp1_po_s = self.state_posterior(o_tp1)
            s_tp1_posterior_dist = Normal(s_tp1_po_m, s_tp1_po_s)

            # s_tp1 = self.state_posterior(o_tp1, sample=True)
            # s_tp1 = s_tp1_prior_dist.rsample()
            s_tp1 = s_tp1_posterior_dist.rsample()
            o_tp1_recon = self.decoder(s_tp1)

            o_recon.append(o_tp1_recon)
            priors.append([s_tp1_pr_m, s_tp1_pr_s])
            posteriors.append([s_tp1_po_m, s_tp1_po_s])

            h_t, s_t = h_tp1, s_tp1
            
        return o_recon, priors, posteriors
    
    def train_step(self, o, optimizer, **kwargs):
        
        device = self.get_device()
        optimizer.zero_grad()
        
        o_recon, priors, posteriors = self(o)

        prior_dist = Normal(*map(torch.stack, zip(*priors)))
        posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
        kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

        recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()

        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()
        
        return {'loss': loss.item()}
        # return {'loss': loss.item(), 'recon_loss': recon_loss.item(), 'kl_loss': kl_loss.item()}
        
    def validation_step(self, o, **kwargs):
        
        device = self.get_device()
        o_recon, priors, posteriors = self(o)
        
        prior_dist = Normal(*map(torch.stack, zip(*priors)))
        posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
        kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

        recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()

        loss = recon_loss + kl_loss
        
        return {'loss': loss.item()}
    
    def eval_step(self, dl, rollout_obs_num=1, **kwargs):
        device = kwargs["device"]
        
        score = []
        recon_ = []
        kl_ = []
        # dyn_error_latent = []
        # dyn_error_image = []
        dyn_error_ = []
        
        for o, _ in dl: 
            o = o.to(device)
            B, C, W, H, T = o.shape
            Z = self.lstate_size
            
            # Mean Condition Number
            o_ = o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H).to(device)
            z, _ = self.state_posterior(o_)
            # G = get_pullbacked_Riemannian_metric(self.decoder, z, create_graph=False)
            # score.append(get_flattening_scores(G, mode='condition_number'))
            
            # Specific loss 
            o_recon, priors, posteriors = self(o)

            prior_dist = Normal(*map(torch.stack, zip(*priors)))
            posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
            kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

            recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
            
            z_sample = posterior_dist.rsample().view(-1, self.lstate_size)
            
            recon_.append(recon_loss)
            kl_.append(kl_loss)
            
            # Rollout Error(Dynamics error)
            lstates_rollout, o_rollout = self.rollout_multiple_observation(o=o[..., :rollout_obs_num], T_rollout=T-rollout_obs_num)
            # lstates, _ = self.state_posterior(o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H))
            # lstates = lstates.view(B, T, Z).permute(0, 2, 1)
            # dyn_error_latent.append(((lstates - lstates_rollout)**2).sum((1, 2)).mean())
            # dyn_error_image.append(((o - o_rollout)**2).sum((1, 2, 3, 4)).mean())
            
            # # ONLY FOR ROTATINGMNIST GROUND TRUTH
            # o_dynamics = torch.zeros_like(o_rollout)
            # for t_idx in range(T):
            #     o_dynamics[..., t_idx] = affine(img=o_rollout[..., 0], angle=10*t_idx, translate=[0, 0], scale=1., shear=0)
            
            # GENERAL o_dynamics
            o_dynamics = o[..., rollout_obs_num:]
            
            dyn_error_.append(((o_rollout - o_dynamics)**2).sum((1, 2, 3, 4)).mean())      
            
        mean_condition_number = torch.cat(score).mean()
        recon_ = torch.stack(recon_).mean()
        kl_ = torch.stack(kl_).mean()
        # dyn_error_latent = torch.stack(dyn_error_latent).mean()
        # dyn_error_image = torch.stack(dyn_error_image).mean()
        dyn_error_ = torch.stack(dyn_error_).mean()
        
        return {
            "eval/MCN_": mean_condition_number.item(),
            "eval/recon_loss_": recon_.item(),
            "eval/kl_loss_": kl_.item(),
            # "eval/dyn_error_latent_": dyn_error_latent.item(),
            # "eval/dyn_error_image_": dyn_error_image.item(),
            "eval/dyn_error_": dyn_error_.item(),
        }
    
    def visualization_step(self, dl, rollout_obs_num=1, **kwargs):
        
        d_val = {}
        
        num_figures = 5
        x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]].to(self.get_device())
        
        lstates, o_recon = self.rollout_multiple_observation(x[..., :rollout_obs_num], dl.dataset.data.shape[-1])
        
        B, C, W, H, T = o_recon.shape
        
        img_grid = torch.cat([x[..., :rollout_obs_num], o_recon], dim=-1) # (B, C, W, H, T+rollout_obs_num)
        img_grid = img_grid.permute(0, 4, 1, 2, 3).reshape(B*(T+rollout_obs_num), C, W, H)
        img_grid = make_grid(img_grid.detach().cpu(), nrow=T+rollout_obs_num, value_range=(0, 1), pad_value=1)
        
        boundary = (W+2)*rollout_obs_num       # boundary line between observation and rollout
        img_grid[0,:,boundary:boundary+2] = 1
        img_grid[1,:,boundary:boundary+2] = 0
        img_grid[2,:,boundary:boundary+2] = 0
        
        d_val['rollout@'] = torch.clip(img_grid, min=0, max=1)
        
        # 2d graph (latent sapce)
        if self.lstate_size == 2:
            num_points_for_each_class = 200
            num_G_plots_for_each_class = 100
            num_rollout_for_each_class = 1
            num_rollout = 36

            label_unique = torch.unique(dl.dataset.targets)

            z_ = []
            z_sampled_ = []
            label_ = []
            label_sampled_ = []
            G_ = []
            z_rollout = []
            o_rollout = []
            label_rollout = []

            for label in label_unique:
                temp_data = dl.dataset.data[dl.dataset.targets == label][:num_points_for_each_class][..., 0]
                temp_z, _ = self.state_posterior(temp_data.to(self.get_device()))
                
                z_sampled = temp_z[torch.randperm(len(temp_z))[:num_G_plots_for_each_class]]
                # G = get_pullbacked_Riemannian_metric(self.decoder, z_sampled)
                z_.append(temp_z)
                label_.append(label.repeat(temp_z.size(0)))
                z_sampled_.append(z_sampled)
                label_sampled_.append(label.repeat(z_sampled.size(0)))
                # G_.append(G)
                
                temp_data = dl.dataset.data[dl.dataset.targets == label][:num_rollout_for_each_class][..., :rollout_obs_num]
                lstates, o_recon = self.rollout_multiple_observation(temp_data.to(self.get_device()), num_rollout)
                z_rollout.append(lstates)
                o_rollout.append(o_recon)
                label_rollout.append(label.repeat(lstates.shape[0]))
                
            z_ = torch.cat(z_, dim=0).detach().cpu().numpy()
            label_ = torch.cat(label_, dim=0).detach().cpu().numpy()
            color_ = label_to_color(label_)
            G_ = torch.cat(G_, dim=0).detach().cpu()
            z_sampled_ = torch.cat(z_sampled_, dim=0).detach().cpu().numpy()
            label_sampled_ = torch.cat(label_sampled_, dim=0).detach().cpu().numpy()
            color_sampled_ = label_to_color(label_sampled_)
            z_rollout = torch.cat(z_rollout, dim=0).detach().cpu().numpy()
            o_rollout = torch.cat(o_rollout, dim=0).detach().cpu().numpy()
            label_rollout = torch.cat(label_rollout, dim=0).detach().cpu().numpy()
            color_rollout = label_to_color(label_rollout)
            
            f = plt.figure(figsize=(10, 10))
            plt.title('Latent space embeddings with equidistant ellipses')
            z_scale = np.minimum(np.max(z_, axis=0), np.min(z_, axis=0))
            eig_mean = torch.svd(G_).S.mean().item()
            scale = 0.1 * z_scale * np.sqrt(eig_mean)
            alpha = 0.3
            for idx in range(len(z_sampled_)):
                e = PD_metric_to_ellipse(np.linalg.inv(G_[idx,:,:]), z_sampled_[idx,:], scale, fc=color_sampled_[idx,:]/255.0, alpha=alpha)
                plt.gca().add_artist(e)
            for label in label_unique:
                label = label.item()
                plt.scatter(z_[label_==label,0], z_[label_==label,1], color=color_[label_==label][0]/255, label=label)
                
            for idx in range(len(z_rollout)):
                plt.plot(z_rollout[idx, 0, :], z_rollout[idx, 1, :], '-o', color='k')
                for t_idx in range(z_rollout.shape[-1]):
                    im = OffsetImage(o_rollout[idx, :, :, :, t_idx].transpose(1, 2, 0), zoom=0.7)
                    if t_idx < rollout_obs_num:             # given observations
                        ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=True)
                    elif t_idx == z_rollout.shape[-1]-1:     # the final frame of rollout
                        ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=True, bboxprops=dict(edgecolor='red'))    
                    else:
                        ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=False)
                    plt.gca().add_artist(ab)
            plt.legend()
            plt.axis('equal')
            plt.close()
            f_np = np.transpose(figure_to_array(f), (2, 0, 1))[:3,:,:]
            
            d_val['latent_space@'] = f_np
        
        # Step size plot of Dynamics Rollout
        num_rollout_for_each_class = 4
        num_rollout = 36

        label_unique = torch.unique(dl.dataset.targets)

        z_ = []
        z_sampled_ = []
        label_ = []
        label_sampled_ = []
        G_ = []
        z_rollout = []
        o_rollout = []
        label_rollout = []

        for label in label_unique:
            temp_data = dl.dataset.data[dl.dataset.targets == label][:num_rollout_for_each_class][..., :rollout_obs_num]
            lstates, o_recon = self.rollout_multiple_observation(temp_data.to(self.get_device()), num_rollout)
            z_rollout.append(lstates)
            o_rollout.append(o_recon)
            label_rollout.append(label.repeat(lstates.shape[0]))
            
        z_rollout = torch.cat(z_rollout, dim=0).detach().cpu()
        o_rollout = torch.cat(o_rollout, dim=0).detach().cpu()
        label_rollout = torch.cat(label_rollout, dim=0).detach().cpu()
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle('Stepsize of Dynamics Rollout', y=0.92, fontsize=20)
        for label in label_unique:
            z_ = z_rollout[label_rollout == label]
            o_ = o_rollout[label_rollout == label]
            for idx in range(4):
                ax_idx = (idx//2, idx%2)
                stepsize_ = (z_[idx, :, 1:] - z_[idx, :, :-1]).norm(dim=0) / np.sqrt(self.lstate_size)
                axs[ax_idx].plot(np.arange(len(stepsize_))+0.5, stepsize_, '-o', label=label.item())
                axs[ax_idx].legend()
        plt.close()
        f_np = np.transpose(figure_to_array(fig), (2, 0, 1))[:3,:,:]
        
        d_val['dyn_stepsize@'] = f_np
        
        return d_val
    
# class IsoRSSM(RecurrentStateSpaceModel):
#     def __init__(self,
#             encoder,
#             decoder,
#             hstate_dim=200,
#             hidden_dim=200,
#             activation='relu',
#             metric='identity',
#             iso_reg=1.0,
#         ):
#         super().__init__(encoder=encoder, 
#                             decoder=decoder, 
#                             hstate_dim=hstate_dim, 
#                             hidden_dim=hidden_dim, 
#                             activation=activation)
        
#         self.metric = metric
#         self.iso_reg = iso_reg
        
#     def train_step(self, o, optimizer, **kwargs):
        
#         device = self.get_device()
#         optimizer.zero_grad()
        
#         o_recon, priors, posteriors = self(o)

#         prior_dist = Normal(*map(torch.stack, zip(*priors)))
#         posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#         kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
        
#         z_sample = posterior_dist.rsample().view(-1, self.lstate_size)
#         # iso_loss = relaxed_distortion_measure(self.decoder, z_sample, eta=0.2, metric=self.metric)

#         loss = recon_loss + kl_loss + self.iso_reg * iso_loss
#         loss.backward()
#         optimizer.step()
        
#         return {'loss': loss.item()}
    
#     def validation_step(self, o, **kwargs):
        
#         device = self.get_device()
        
#         o_recon, priors, posteriors = self(o)

#         prior_dist = Normal(*map(torch.stack, zip(*priors)))
#         posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#         kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
        
#         z_sample = posterior_dist.rsample().view(-1, self.lstate_size)
#         iso_loss = relaxed_distortion_measure(self.decoder, z_sample, eta=0.2, metric=self.metric)
        
#         loss = recon_loss + kl_loss + self.iso_reg * iso_loss
        
#         return {'loss': loss.item()}
    
#     def eval_step(self, dl, rollout_obs_num=1, **kwargs):
#         device = kwargs["device"]
        
#         score = []
#         recon_ = []
#         kl_ = []
#         iso_ = []
#         # dyn_error_latent = []
#         # dyn_error_image = []
#         dyn_error_ = []
        
#         for o, _ in dl: 
#             o = o.to(device)
#             B, C, W, H, T = o.shape
#             Z = self.lstate_size
            
#             # Mean Condition Number
#             o_ = o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H).to(device)
#             z, _ = self.state_posterior(o_)
#             G = get_pullbacked_Riemannian_metric(self.decoder, z, create_graph=False)
#             score.append(get_flattening_scores(G, mode='condition_number'))
            
#             # Specific loss 
#             o_recon, priors, posteriors = self(o)

#             prior_dist = Normal(*map(torch.stack, zip(*priors)))
#             posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#             kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

#             recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
            
#             z_sample = posterior_dist.rsample().view(-1, self.lstate_size)
#             iso_loss = relaxed_distortion_measure(self.decoder, z_sample, eta=0.2, metric=self.metric)
            
#             recon_.append(recon_loss)
#             kl_.append(kl_loss)
#             iso_.append(iso_loss)
            
#             # Rollout Error(Dynamics error)
#             lstates_rollout, o_rollout = self.rollout_multiple_observation(o=o[..., :rollout_obs_num], T_rollout=T-rollout_obs_num)
#             # lstates, _ = self.state_posterior(o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H))
#             # lstates = lstates.view(B, T, Z).permute(0, 2, 1)
#             # dyn_error_latent.append(((lstates - lstates_rollout)**2).sum((1, 2)).mean())
#             # dyn_error_image.append(((o - o_rollout)**2).sum((1, 2, 3, 4)).mean())
            
#             # # ONLY FOR ROTATINGMNIST GROUND TRUTH
#             # o_dynamics = torch.zeros_like(o_rollout)
#             # for t_idx in range(T):
#             #     o_dynamics[..., t_idx] = affine(img=o_rollout[..., 0], angle=10*t_idx, translate=[0, 0], scale=1., shear=0)
            
#             # GENERAL o_dynamics
#             o_dynamics = o[..., rollout_obs_num:]
#             dyn_error_.append(((o_rollout - o_dynamics)**2).sum((1, 2, 3, 4)).mean())    
        
#         mean_condition_number = torch.cat(score).mean()
#         recon_ = torch.stack(recon_).mean()
#         kl_ = torch.stack(kl_).mean()
#         iso_ = torch.stack(iso_).mean()
#         # dyn_error_latent = torch.stack(dyn_error_latent).mean()
#         # dyn_error_image = torch.stack(dyn_error_image).mean()
#         dyn_error_ = torch.stack(dyn_error_).mean()
        
#         return {
#             "eval/MCN_": mean_condition_number.item(),
#             "eval/recon_loss_": recon_.item(),
#             "eval/kl_loss_": kl_.item(),
#             "eval/iso_loss_": iso_.item(),
#             # "eval/dyn_error_latent_": dyn_error_latent.item(),
#             # "eval/dyn_error_image_": dyn_error_image.item(),
#             "eval/dyn_error_": dyn_error_.item(),
#         }
        
# class NRRSSM(RecurrentStateSpaceModel):
#     def __init__(self,
#             encoder,
#             decoder,
#             hstate_dim=200,
#             hidden_dim=200,
#             activation='relu',
#             k=4,
#             approx_order=1,
#         ):
#         super().__init__(encoder=encoder, 
#                             decoder=decoder, 
#                             hstate_dim=hstate_dim, 
#                             hidden_dim=hidden_dim, 
#                             activation=activation)
        
#         self.k = k
#         self.approx_order = approx_order
        
#     def train_step(self, o, optimizer, **kwargs):
        
#         device = self.get_device()
#         optimizer.zero_grad()
#         B, C, W, H, T = o.shape
#         Z = self.lstate_size
#         k = self.k
        
#         o_recon, priors, posteriors = self(o)

#         prior_dist = Normal(*map(torch.stack, zip(*priors)))
#         posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#         kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
        
#         o_nn = torch.zeros_like(o).unsqueeze(-1).repeat_interleave(k, dim=-1)
#         for t_idx in range(T):
#             nn_idx = range(max(t_idx-int(k/2), 0), max(t_idx-int(k/2), 0)+k+1) if max(t_idx-int(k/2), 0)+k < T else range(min(t_idx+int(k/2), T-1)-k, min(t_idx+int(k/2), T-1)+1)
#             nn_idx = list(nn_idx)
#             nn_idx.remove(t_idx)
#             o_nn[..., t_idx, :] = o[..., nn_idx]
            
#         o_ = o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H)
#         o_nn_ = o_nn.permute(0, 4, 5, 1, 2, 3).reshape(B*T, k, C, W, H)
#         lstate, _ = self.state_posterior(o_)
#         lstate_nn, _ = self.state_posterior(o_nn_.view(B*T*k, C, W, H))
#         lstate_nn = lstate_nn.view(B*T, k, Z)
#         o_nn_recon = self.neighborhood_recon(lstate, lstate_nn)
        
#         nn_recon_loss = ((o_nn_recon - o_nn_)**2).sum((2, 3, 4)).mean()

#         loss = recon_loss + kl_loss + nn_recon_loss
#         loss.backward()
#         optimizer.step()
        
#         return {'loss': loss.item()}
    
#     def validation_step(self, o, **kwargs):
        
#         device = self.get_device()
        
#         B, C, W, H, T = o.shape
#         Z = self.lstate_size
#         k = self.k
        
#         o_recon, priors, posteriors = self(o)

#         prior_dist = Normal(*map(torch.stack, zip(*priors)))
#         posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#         kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
        
#         o_nn = torch.zeros_like(o).unsqueeze(-1).repeat_interleave(k, dim=-1)
#         for t_idx in range(T):
#             nn_idx = range(max(t_idx-int(k/2), 0), max(t_idx-int(k/2), 0)+k+1) if max(t_idx-int(k/2), 0)+k < T else range(min(t_idx+int(k/2), T-1)-k, min(t_idx+int(k/2), T-1)+1)
#             nn_idx = list(nn_idx)
#             nn_idx.remove(t_idx)
#             o_nn[..., t_idx, :] = o[..., nn_idx]
            
#         o_ = o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H)
#         o_nn_ = o_nn.permute(0, 4, 5, 1, 2, 3).reshape(B*T, k, C, W, H)
#         lstate, _ = self.state_posterior(o_)
#         lstate_nn, _ = self.state_posterior(o_nn_.view(B*T*k, C, W, H))
#         lstate_nn = lstate_nn.view(B*T, k, Z)
#         o_nn_recon = self.neighborhood_recon(lstate, lstate_nn)
        
#         nn_recon_loss = ((o_nn_recon - o_nn_)**2).sum((2, 3, 4)).mean()

#         loss = recon_loss + kl_loss + nn_recon_loss
        
#         return {'loss': loss.item()}
    
#     def eval_step(self, dl, rollout_obs_num=1, **kwargs):
#         device = kwargs["device"]
        
#         # score = []
#         recon_ = []
#         kl_ = []
#         nr_ = []
#         # dyn_error_latent = []
#         # dyn_error_image = []
#         dyn_error_ = []
        
#         for o, _ in dl: 
#             o = o.to(device)
#             B, C, W, H, T = o.shape
#             Z = self.lstate_size
#             k = self.k
            
#             # # Mean Condition Number
#             # o_ = o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H).to(device)
#             # z, _ = self.state_posterior(o_)
#             # G = get_pullbacked_Riemannian_metric(self.decoder, z, create_graph=False)
#             # score.append(get_flattening_scores(G, mode='condition_number'))
            
#             # Specific loss 
#             o_recon, priors, posteriors = self(o)

#             prior_dist = Normal(*map(torch.stack, zip(*priors)))
#             posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#             kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

#             recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
            
#             o_nn = torch.zeros_like(o).unsqueeze(-1).repeat_interleave(k, dim=-1)
#             for t_idx in range(T):
#                 nn_idx = range(max(t_idx-int(k/2), 0), max(t_idx-int(k/2), 0)+k+1) if max(t_idx-int(k/2), 0)+k < T else range(min(t_idx+int(k/2), T-1)-k, min(t_idx+int(k/2), T-1)+1)
#                 nn_idx = list(nn_idx)
#                 nn_idx.remove(t_idx)
#                 o_nn[..., t_idx, :] = o[..., nn_idx]
                
#             o_ = o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H)
#             o_nn_ = o_nn.permute(0, 4, 5, 1, 2, 3).reshape(B*T, k, C, W, H)
#             lstate, _ = self.state_posterior(o_)
#             lstate_nn, _ = self.state_posterior(o_nn_.view(B*T*k, C, W, H))
#             lstate_nn = lstate_nn.view(B*T, k, Z)
#             o_nn_recon = self.neighborhood_recon(lstate, lstate_nn)
            
#             nn_recon_loss = ((o_nn_recon - o_nn_)**2).sum((2, 3, 4)).mean()
            
#             recon_.append(recon_loss)
#             kl_.append(kl_loss)
#             nr_.append(nn_recon_loss)
            
#             # Rollout Error(Dynamics error)
#             lstates_rollout, o_rollout = self.rollout_multiple_observation(o=o[..., :rollout_obs_num], T_rollout=T-rollout_obs_num)
#             # lstates, _ = self.state_posterior(o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H))
#             # lstates = lstates.view(B, T, Z).permute(0, 2, 1)
#             # dyn_error_latent.append(((lstates - lstates_rollout)**2).sum((1, 2)).mean())
#             # dyn_error_image.append(((o - o_rollout)**2).sum((1, 2, 3, 4)).mean())
            
#             # # ONLY FOR ROTATINGMNIST GROUND TRUTH
#             # o_dynamics = torch.zeros_like(o_rollout)
#             # for t_idx in range(T):
#             #     o_dynamics[..., t_idx] = affine(img=o_rollout[..., 0], angle=10*t_idx, translate=[0, 0], scale=1., shear=0)
            
#             # GENERAL o_dynamics
#             o_dynamics = o[..., rollout_obs_num:]
#             dyn_error_.append(((o_rollout - o_dynamics)**2).sum((1, 2, 3, 4)).mean()) 
        
#         # mean_condition_number = torch.cat(score).mean()
#         recon_ = torch.stack(recon_).mean()
#         kl_ = torch.stack(kl_).mean()
#         nr_ = torch.stack(nr_).mean()
#         # dyn_error_latent = torch.stack(dyn_error_latent).mean()
#         # dyn_error_image = torch.stack(dyn_error_image).mean()
#         dyn_error_ = torch.stack(dyn_error_).mean()
        
#         return {
#             # "eval/MCN_": mean_condition_number.item(),
#             "eval/recon_loss_": recon_.item(),
#             "eval/kl_loss_": kl_.item(),
#             "eval/nr_loss_": nr_.item(),
#             # "eval/dyn_error_latent_": dyn_error_latent.item(),
#             # "eval/dyn_error_image_": dyn_error_image.item(),
#             "eval/dyn_error_": dyn_error_.item(),
#         }
    
#     def jacobian(self, z, dz, create_graph=True):
#         # z : (B*T, Z), dz : (B*T, k, Z)
#         batch_size = dz.shape[0]
#         num_nn = dz.shape[1]
#         z_dim = self.lstate_size

#         v = dz.view(-1, z_dim)  # (bs * num_nn , z_dim)
#         inputs = (z.unsqueeze(1).repeat(1, num_nn, 1).view(-1, z_dim))  # (bs * num_nn , z_dim)
#         jac = torch.autograd.functional.jvp(self.decoder, inputs, v=v, create_graph=create_graph)[1]
#         jac = jac.view(batch_size, num_nn, *jac.shape[1:])
        
#         return jac
    
#     def jacobian_and_hessian(self, z, dz, create_graph=True):
#         # z : (B*T, Z), dz : (B*T, k, Z)
#         batch_size = dz.shape[0]
#         num_nn = dz.shape[1]
#         z_dim = self.lstate_size

#         v = dz.view(-1, z_dim)  # (B*T*k , z_dim)
#         inputs = (z.unsqueeze(1).repeat(1, num_nn, 1).view(-1, z_dim))  # (B*T*k , z_dim)

#         def jac_temp(inputs):
#             jac = torch.autograd.functional.jvp(self.decoder, inputs, v=v, create_graph=create_graph)[1]
#             return jac

#         temp = torch.autograd.functional.jvp(jac_temp, inputs, v=v, create_graph=create_graph)

#         jac = temp[0].view(batch_size, num_nn, *temp[0].shape[1:])
#         hessian = temp[1].view(batch_size, num_nn, *temp[1].shape[1:])
#         return jac, hessian
    
#     def neighborhood_recon(self, lstate, lstate_nn):
#         # lstate : (B*T, Z), lstate_nn : (B*T, k, Z)
#         o_recon = self.decoder(lstate)
        
#         o_recon = o_recon.unsqueeze(1)  # (B*T, 1, C, W, H)
#         d_lstate = lstate_nn - lstate.unsqueeze(1)  # (B*T, k, z_dim)
        
#         if self.approx_order == 1:
#             Jdz = self.jacobian(lstate, d_lstate)  # (B*T, k, x_dim)
#             o_recon_nn = o_recon + Jdz
#         elif self.approx_order == 2:
#             Jdz, dzHdz = self.jacobian_and_hessian(lstate, d_lstate)
#             o_recon_nn = o_recon + Jdz + 0.5*dzHdz
#         return o_recon_nn
    
# class ERRSSM(RecurrentStateSpaceModel):
#     def __init__(self,
#             encoder,
#             decoder,
#             hstate_dim=200,
#             hidden_dim=200,
#             activation='relu',
#             iso_reg=1.0,
#             bdry_reg=0.0,
#             measure='irvae',
#             bdry_measure='time_bdry',
#             distfunc = 'TimeSteps',
#             bandwidth = 2
#         ):
#         super().__init__(encoder=encoder, 
#                             decoder=decoder, 
#                             hstate_dim=hstate_dim, 
#                             hidden_dim=hidden_dim, 
#                             activation=activation)
        
#         self.iso_reg = iso_reg
#         self.bdry_reg = bdry_reg
#         self.measure = measure
#         self.bdry_measure = bdry_measure
#         self.distfunc = distfunc
#         self.bandwidth = bandwidth
        
#         self.c = 1/4
        
#     def train_step(self, o, optimizer, **kwargs):
#         """
#         o: (B, *dims, T)
#         o_L: (B, T, *dims)
#         """
#         device = self.get_device()
#         optimizer.zero_grad()
        
#         # compute dynamics learning loss
#         o_recon, priors, posteriors = self(o)
#         prior_dist = Normal(*map(torch.stack, zip(*priors)))
#         posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#         kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()
#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
        
#         # compute embedding (to compute pushforward metric)
#         _, z_0 = self.get_init_state(o[..., 0], sample=False)
#         posterior_mean = torch.stack([m for [m,s] in posteriors], dim=0)
#         z = torch.cat([z_0.unsqueeze(0), posterior_mean], dim=0).transpose(0,1) # mean instead of sample
#         # z_sample = torch.cat([z_0.unsqueeze(0), posterior_dist.rsample()], dim=0).transpose(0, 1)
        
#         # compute iso loss and boundary loss
#         o_L = o.unsqueeze(1).transpose(1,-1).flatten(2,-1)
#         L = get_laplacian(o_L, torch.zeros(o_L.shape[:2]), distfunc='TimeSteps', bandwidth=self.bandwidth)
#         H_tilde = get_JGinvJT(L, z)
#         iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde, eta=0.2, measure=self.measure)
#         bdry_loss = boundary_loss(o_L, z, bdry_measure=self.bdry_measure)
        
#         # compute and optimize total loss
#         if self.measure == 'irvae':
#             loss = (recon_loss + kl_loss) + self.iso_reg * iso_loss
#         elif self.measure == 'harmonic':
#             loss = (recon_loss + kl_loss) - self.iso_reg * self.bdry_reg * bdry_loss
#         else:
#             raise NotImplementedError
#         loss.backward()
#         optimizer.step()

#         return {
#             'loss': loss.item(),
#             'loss/train_recon_loss_': recon_loss.item(),
#             'loss/train_kl_loss_': kl_loss.item(),
#             'loss/train_iso_loss_': iso_loss.item(),
#         }
    
#     def validation_step(self, o, **kwargs):
#         device = self.get_device()
        
#         # compute dynamics learning loss
#         o_recon, priors, posteriors = self(o)
#         prior_dist = Normal(*map(torch.stack, zip(*priors)))
#         posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#         kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()
#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
        
#         # compute embedding (to compute pushforward metric)
#         _, z_0 = self.get_init_state(o[..., 0], sample=False)
#         posterior_mean = torch.stack([m for [m,s] in posteriors], dim=0)
#         z = torch.cat([z_0.unsqueeze(0), posterior_mean], dim=0).transpose(0,1) # mean instead of sample
#         # z_sample = torch.cat([z_0.unsqueeze(0), posterior_dist.rsample()], dim=0).transpose(0, 1)
        
#         # compute iso loss and boundary loss
#         o_L = o.unsqueeze(1).transpose(1,-1).flatten(2,-1)
#         L = get_laplacian(o_L, torch.zeros(o_L.shape[:2]), distfunc='TimeSteps', bandwidth=self.bandwidth)
#         H_tilde = get_JGinvJT(L, z)
#         iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde, eta=0.2, measure=self.measure)
#         bdry_loss = boundary_loss(o_L, z, bdry_measure=self.bdry_measure)
        
#         # compute total loss
#         if self.measure == 'irvae':
#             loss = (recon_loss + kl_loss) + self.iso_reg * iso_loss
#         elif self.measure == 'harmonic':
#             loss = (recon_loss + kl_loss) - self.iso_reg * self.bdry_reg * bdry_loss
#         else:
#             raise NotImplementedError

#         return {
#             'loss': loss.item(),
#             'loss/train_recon_loss_': recon_loss.item(),
#             'loss/train_kl_loss_': kl_loss.item(),
#             'loss/train_iso_loss_': iso_loss.item(),
#         }
    
#     def eval_step(self, dl, rollout_obs_num=1, **kwargs):
#         device = kwargs["device"]
        
#         mcn_pb = []
#         mcn_pf = []
#         recon_ = []
#         kl_ = []
#         dyn_error_ = []
#         iso_ = []
        
#         for o, _ in dl: 
#             o = o.to(device)
#             B, C, W, H, T = o.shape
#             Z = self.lstate_size
            
#             # Mean Condition Number (Pullback)
#             o_ = o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H).to(device)
#             z, _ = self.state_posterior(o_)
#             G = get_pullbacked_Riemannian_metric(self.decoder, z, create_graph=False)
#             mcn_pb.append(get_flattening_scores(G, mode='condition_number'))
            
#             # Specific loss 
#             o_recon, priors, posteriors = self(o)

#             prior_dist = Normal(*map(torch.stack, zip(*priors)))
#             posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#             kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

#             recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
            
#             z_sample = posterior_dist.rsample().view(-1, self.lstate_size)
            
#             recon_.append(recon_loss)
#             kl_.append(kl_loss)
            
#             # Rollout Error(Dynamics error)
#             lstates_rollout, o_rollout = self.rollout_multiple_observation(o=o[..., :rollout_obs_num], T_rollout=T-rollout_obs_num)
            
#             # GENERAL o_dynamics
#             o_dynamics = o[..., rollout_obs_num:]
            
#             dyn_error_.append(((o_rollout - o_dynamics)**2).sum((1, 2, 3, 4)).mean())  
            
#             _, z_0 = self.get_init_state(o[..., 0], sample=False)
#             z_sample = torch.cat([z_0.unsqueeze(0), posterior_dist.rsample()], dim=0).transpose(0, 1)
            
#             B, T, *dims = z_sample.shape
#             dist_XX = torch.zeros(T, T)
#             for i_idx in range(T):
#                 dist_XX[i_idx, :] = torch.abs(torch.arange(T) - i_idx)
#             dist_XX = dist_XX.unsqueeze(0).repeat_interleave(B, dim=0).to(device)
            
#             K = torch.exp(-dist_XX**2 / self.bandwidth**2)
#             d_i = K.sum(dim=1)
#             D_inv = torch.diag_embed(1/d_i)
            
#             K_tilde = D_inv @ K @ D_inv
#             D_tilde_inv = torch.diag_embed(1/K_tilde.sum(dim=1))
#             L = (D_tilde_inv@K_tilde - torch.diag_embed(torch.ones(B, T)).to(device))/(self.c*self.bandwidth)
            
#             H_tilde = get_JGinvJT(L, z_sample)
#             iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde, eta=0.2, measure=self.measure)   
            
#             iso_.append(iso_loss)
#             mcn_pf.append(get_flattening_scores(H_tilde.flatten(0, 1), mode='condition_number'))
                
#         mcn_pb = torch.cat(mcn_pb).mean()
#         mcn_pf = torch.cat(mcn_pf).mean()
#         recon_ = torch.stack(recon_).mean()
#         kl_ = torch.stack(kl_).mean()
#         dyn_error_ = torch.stack(dyn_error_).mean()
#         iso_ = torch.stack(iso_).mean()
            
#         return {
#             "eval/MCN_pullback_": mcn_pb.item(),
#             "eval/MCN_pushforward_": mcn_pf.item(),
#             "eval/recon_loss_": recon_.item(),
#             "eval/kl_loss_": kl_.item(),
#             "eval/dyn_error_": dyn_error_.item(),
#             "eval/iso_loss_": iso_.item(),
#         }
        
#     def visualization_step(self, dl, rollout_obs_num=1, **kwargs):
        
#         d_val = {}
        
#         num_figures = 5
#         x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]].to(self.get_device())
        
#         lstates, o_recon = self.rollout_multiple_observation(x[..., :rollout_obs_num], dl.dataset.data.shape[-1])
        
#         B, C, W, H, T = o_recon.shape
        
#         img_grid = torch.cat([x[..., :rollout_obs_num], o_recon], dim=-1) # (B, C, W, H, T+rollout_obs_num)
#         img_grid = img_grid.permute(0, 4, 1, 2, 3).reshape(B*(T+rollout_obs_num), C, W, H)
#         img_grid = make_grid(img_grid.detach().cpu(), nrow=T+rollout_obs_num, value_range=(0, 1), pad_value=1)
        
#         boundary = (W+2)*rollout_obs_num       # boundary line between observation and rollout
#         img_grid[0,:,boundary:boundary+2] = 1
#         img_grid[1,:,boundary:boundary+2] = 0
#         img_grid[2,:,boundary:boundary+2] = 0
        
#         d_val['rollout@'] = torch.clip(img_grid, min=0, max=1)
        
#         # 2d graph (latent sapce)
#         if self.lstate_size == 2:
#             num_points_for_each_class = 200
#             num_G_plots_for_each_class = 100
#             num_rollout_for_each_class = 1
#             num_rollout = 36

#             label_unique = torch.unique(dl.dataset.targets)

#             z_ = []
#             z_sampled_ = []
#             label_ = []
#             label_sampled_ = []
#             G_ = []
#             z_rollout = []
#             o_rollout = []
#             label_rollout = []

#             for label in label_unique:
#                 temp_data = dl.dataset.data[dl.dataset.targets == label][:num_points_for_each_class][..., 0]
#                 temp_z, _ = self.state_posterior(temp_data.to(self.get_device()))
                
#                 z_sampled = temp_z[torch.randperm(len(temp_z))[:num_G_plots_for_each_class]]
#                 G = get_pullbacked_Riemannian_metric(self.decoder, z_sampled)
#                 z_.append(temp_z)
#                 label_.append(label.repeat(temp_z.size(0)))
#                 z_sampled_.append(z_sampled)
#                 label_sampled_.append(label.repeat(z_sampled.size(0)))
#                 G_.append(G)
                
#                 temp_data = dl.dataset.data[dl.dataset.targets == label][:num_rollout_for_each_class][..., :rollout_obs_num]
#                 lstates, o_recon = self.rollout_multiple_observation(temp_data.to(self.get_device()), num_rollout)
#                 z_rollout.append(lstates)
#                 o_rollout.append(o_recon)
#                 label_rollout.append(label.repeat(lstates.shape[0]))
                
#             z_ = torch.cat(z_, dim=0).detach().cpu().numpy()
#             label_ = torch.cat(label_, dim=0).detach().cpu().numpy()
#             color_ = label_to_color(label_)
#             G_ = torch.cat(G_, dim=0).detach().cpu()
#             z_sampled_ = torch.cat(z_sampled_, dim=0).detach().cpu().numpy()
#             label_sampled_ = torch.cat(label_sampled_, dim=0).detach().cpu().numpy()
#             color_sampled_ = label_to_color(label_sampled_)
#             z_rollout = torch.cat(z_rollout, dim=0).detach().cpu().numpy()
#             o_rollout = torch.cat(o_rollout, dim=0).detach().cpu().numpy()
#             label_rollout = torch.cat(label_rollout, dim=0).detach().cpu().numpy()
#             color_rollout = label_to_color(label_rollout)
            
#             f = plt.figure(figsize=(10, 10))
#             plt.title('Latent space embeddings with equidistant ellipses')
#             z_scale = np.minimum(np.max(z_, axis=0), np.min(z_, axis=0))
#             eig_mean = torch.svd(G_).S.mean().item()
#             scale = 0.1 * z_scale * np.sqrt(eig_mean)
#             alpha = 0.3
#             for idx in range(len(z_sampled_)):
#                 e = PD_metric_to_ellipse(np.linalg.inv(G_[idx,:,:]), z_sampled_[idx,:], scale, fc=color_sampled_[idx,:]/255.0, alpha=alpha)
#                 plt.gca().add_artist(e)
#             for label in label_unique:
#                 label = label.item()
#                 plt.scatter(z_[label_==label,0], z_[label_==label,1], color=color_[label_==label][0]/255, label=label)
                
#             for idx in range(len(z_rollout)):
#                 plt.plot(z_rollout[idx, 0, :], z_rollout[idx, 1, :], '-o', color='k')
#                 for t_idx in range(z_rollout.shape[-1]):
#                     im = OffsetImage(o_rollout[idx, :, :, :, t_idx].transpose(1, 2, 0), zoom=0.7)
#                     if t_idx < rollout_obs_num:             # given observations
#                         ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=True)
#                     elif t_idx == z_rollout.shape[-1]-1:     # the final frame of rollout
#                         ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=True, bboxprops=dict(edgecolor='red'))    
#                     else:
#                         ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=False)
#                     plt.gca().add_artist(ab)
#             plt.legend()
#             plt.axis('equal')
#             plt.close()
#             f_np = np.transpose(figure_to_array(f), (2, 0, 1))[:3,:,:]
            
#             d_val['latent_space@'] = f_np
        
#         # Step size plot of Dynamics Rollout
#         num_rollout_for_each_class = 4
#         num_rollout = 36

#         label_unique = torch.unique(dl.dataset.targets)

#         z_ = []
#         z_sampled_ = []
#         label_ = []
#         label_sampled_ = []
#         G_ = []
#         z_rollout = []
#         o_rollout = []
#         label_rollout = []

#         for label in label_unique:
#             temp_data = dl.dataset.data[dl.dataset.targets == label][:num_rollout_for_each_class][..., :rollout_obs_num]
#             lstates, o_recon = self.rollout_multiple_observation(temp_data.to(self.get_device()), num_rollout)
#             z_rollout.append(lstates)
#             o_rollout.append(o_recon)
#             label_rollout.append(label.repeat(lstates.shape[0]))
            
#         z_rollout = torch.cat(z_rollout, dim=0).detach().cpu()
#         o_rollout = torch.cat(o_rollout, dim=0).detach().cpu()
#         label_rollout = torch.cat(label_rollout, dim=0).detach().cpu()
        
#         fig, axs = plt.subplots(2, 2, figsize=(16, 8))
#         fig.suptitle('Stepsize of Dynamics Rollout', y=0.92, fontsize=20)
#         for label in label_unique:
#             z_ = z_rollout[label_rollout == label]
#             o_ = o_rollout[label_rollout == label]
#             for idx in range(4):
#                 ax_idx = (idx//2, idx%2)
#                 stepsize_ = (z_[idx, :, 1:] - z_[idx, :, :-1]).norm(dim=0) / np.sqrt(self.lstate_size)
#                 axs[ax_idx].plot(np.arange(len(stepsize_))+0.5, stepsize_, '-o', label=label.item())
#                 axs[ax_idx].legend()
#         plt.close()
#         f_np = np.transpose(figure_to_array(fig), (2, 0, 1))[:3,:,:]
        
#         d_val['dyn_stepsize@'] = f_np
        
#         return d_val

# class ERRSSM_RobotArm(RecurrentStateSpaceModel):
#     def __init__(self,
#             encoder,
#             decoder,
#             hstate_dim=200,
#             hidden_dim=200,
#             activation='relu',
#             iso_reg=1.0,
#             measure='irvae',
#             bandwidth=1
#         ):
#         super().__init__(
#             encoder=encoder, 
#             decoder=decoder, 
#             hstate_dim=hstate_dim, 
#             hidden_dim=hidden_dim, 
#             activation=activation
#         )
        
#         self.iso_reg = iso_reg
#         self.measure = measure
#         self.bandwidth = bandwidth
        
#     def train_step(self, o, y, optimizer, **kwargs):
#         """
#         o: (B, *dims, T)
#         y: latent vector (x, y coord. of cube), (B, 2, T)
#         """
#         device = self.get_device()
#         optimizer.zero_grad()
        
#         # compute dynamics learning loss
#         o_recon, priors, posteriors = self(o)
#         prior_dist = Normal(*map(torch.stack, zip(*priors)))
#         posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#         kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()
#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
#         # recon_loss = torch.nn.MSELoss()(o, torch.stack(o_recon, dim=-1))
        
#         # compute embedding (to compute pushforward metric)
#         _, z_0 = self.get_init_state(o[..., 0], sample=False)
#         posterior_mean = torch.stack([m for [m,s] in posteriors], dim=0)
#         z = torch.cat([z_0.unsqueeze(0), posterior_mean], dim=0).transpose(0,1) # mean instead of sample, (B, T, zdim)
#         # z_sample = torch.cat([z_0.unsqueeze(0), posterior_dist.rsample()], dim=0).transpose(0, 1)
        
#         z = z.flatten(0, 1) # (B*T, zdim)
#         y = y.transpose(1, 2).flatten(0, 1) # (B*T, 2)
        
#         # compute iso loss and boundary loss
#         L = get_laplacian(y.unsqueeze(0), torch.zeros(1, y.shape[0]), distfunc='Euclidean', bandwidth=self.bandwidth) # (1, B*T, B*T)
#         H_tilde = get_JGinvJT(L, z.unsqueeze(0)) # (1, B*T, zdim, zdim)
#         iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde, eta=0.2, measure=self.measure)
        
#         # compute and optimize total loss
#         loss = recon_loss + kl_loss + self.iso_reg * iso_loss
#         loss.backward()
#         optimizer.step()

#         return {
#             'loss': loss.item(),
#             'loss/train_recon_loss_': recon_loss.item(),
#             'loss/train_kl_loss_': kl_loss.item(),
#             'loss/train_iso_loss_': iso_loss.item(),
#         }
    
#     def validation_step(self, o, y, **kwargs):
#         device = self.get_device()
        
#         # compute dynamics learning loss
#         o_recon, priors, posteriors = self(o)
#         prior_dist = Normal(*map(torch.stack, zip(*priors)))
#         posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#         kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()
#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
#         # recon_loss = torch.nn.MSELoss()(o, torch.stack(o_recon, dim=-1))
        
#         # compute embedding (to compute pushforward metric)
#         _, z_0 = self.get_init_state(o[..., 0], sample=False)
#         posterior_mean = torch.stack([m for [m,s] in posteriors], dim=0)
#         z = torch.cat([z_0.unsqueeze(0), posterior_mean], dim=0).transpose(0,1) # mean instead of sample, (B, T, zdim)
#         # z_sample = torch.cat([z_0.unsqueeze(0), posterior_dist.rsample()], dim=0).transpose(0, 1)
        
#         z = z.flatten(0, 1) # (B*T, zdim)
#         y = y.transpose(1, 2).flatten(0, 1) # (B*T, 2)
        
#         # compute iso loss and boundary loss
#         L = get_laplacian(y.unsqueeze(0), torch.zeros(1, y.shape[0]), distfunc='Euclidean', bandwidth=self.bandwidth) # (1, B*T, B*T)
#         H_tilde = get_JGinvJT(L, z.unsqueeze(0)) # (1, B*T, zdim, zdim)
#         iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde, eta=0.2, measure=self.measure)
        
#         # compute and optimize total loss
#         loss = recon_loss + kl_loss + self.iso_reg * iso_loss
        
#         return {
#             'loss': loss.item(),
#             'loss/valid_recon_loss_': recon_loss.item(),
#             'loss/valid_kl_loss_': kl_loss.item(),
#             'loss/valid_iso_loss_': iso_loss.item(),
#         }
    
#     def eval_step(self, dl, rollout_obs_num=1, **kwargs):
#         device = kwargs["device"]
        
#         recon_ = []
#         kl_ = []
#         dyn_error_ = []
        
#         for o, _ in dl: 
#             o = o.to(device)
#             B, C, W, H, T = o.shape
#             Z = self.lstate_size
            
#             # Mean Condition Number
#             o_ = o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H).to(device)
#             z, _ = self.state_posterior(o_)
            
#             # Specific loss 
#             o_recon, priors, posteriors = self(o)

#             prior_dist = Normal(*map(torch.stack, zip(*priors)))
#             posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#             kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

#             recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
#             # recon_loss = torch.nn.MSELoss()(o, torch.stack(o_recon, dim=-1))
            
#             z_sample = posterior_dist.rsample().view(-1, self.lstate_size)
            
#             recon_.append(recon_loss)
#             kl_.append(kl_loss)
            
#             # Rollout Error(Dynamics error)
#             lstates_rollout, o_rollout = self.rollout_multiple_observation(o=o[..., :rollout_obs_num], T_rollout=T-rollout_obs_num)
            
#             # GENERAL o_dynamics
#             o_dynamics = o[..., rollout_obs_num:]
            
#             dyn_error_.append(((o_rollout - o_dynamics)**2).sum((1, 2, 3, 4)).mean())  
#             # dyn_error_.append(torch.nn.MSELoss()(o_rollout, o_dynamics))
            
            
#         recon_ = torch.stack(recon_).mean()
#         kl_ = torch.stack(kl_).mean()
#         dyn_error_ = torch.stack(dyn_error_).mean()
        
#         return {
#             "eval/recon_loss_": recon_.item(),
#             "eval/kl_loss_": kl_.item(),
#             "eval/dyn_error_": dyn_error_.item(),
#         }
        
#     def visualization_step(self, dl, rollout_obs_num=1, **kwargs):
#         d_vis = {}
        
#         num_figures = 5
#         x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]].to(self.get_device())
        
#         lstates, o_recon = self.rollout_multiple_observation(x[..., :rollout_obs_num], dl.dataset.data.shape[-1])
        
#         B, C, W, H, T = o_recon.shape
        
#         img_grid = torch.cat([x[..., :rollout_obs_num], o_recon], dim=-1) # (B, C, W, H, T+rollout_obs_num)
#         img_grid = img_grid.permute(0, 4, 1, 2, 3).reshape(B*(T+rollout_obs_num), C, W, H)
#         img_grid = make_grid(img_grid.detach().cpu(), nrow=T+rollout_obs_num, value_range=(0, 1), pad_value=1)
        
#         boundary = (W+2)*rollout_obs_num       # boundary line between observation and rollout
#         img_grid[0,:,boundary:boundary+2] = 1
#         img_grid[1,:,boundary:boundary+2] = 0
#         img_grid[2,:,boundary:boundary+2] = 0
        
#         d_vis['rollout@'] = torch.clip(img_grid, min=0, max=1)
        
#         return d_vis
    
# class TNCRSSM(RecurrentStateSpaceModel):
#     def __init__(self,
#             encoder,
#             decoder,
#             hstate_dim=200,
#             hidden_dim=200,
#             activation='relu',
#             tnc_reg=0,
#             bandwidth=1,
#             neighbor_dist=3,
#         ):
#         super().__init__(encoder=encoder, 
#                             decoder=decoder, 
#                             hstate_dim=hstate_dim, 
#                             hidden_dim=hidden_dim, 
#                             activation=activation,
#         )
        
#         self.tnc_reg = tnc_reg
#         self.neighbor_dist = neighbor_dist
        
#     def train_step(self, o, optimizer, **kwargs):
#         device = self.get_device()
#         optimizer.zero_grad()
        
#         o_recon, priors, posteriors = self(o)

#         prior_dist = Normal(*map(torch.stack, zip(*priors)))
#         posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#         kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
        
#         _, z_0 = self.get_init_state(o[..., 0], sample=False)
#         z_sample = torch.cat([z_0.unsqueeze(0), posterior_dist.rsample()], dim=0).transpose(0, 1)
        
#         B, T, *dims = z_sample.shape
#         dist_XX = torch.zeros(T, T)
#         for i_idx in range(T):
#             dist_XX[i_idx, :] = torch.abs(torch.arange(T) - i_idx)
#         dist_XX = dist_XX.unsqueeze(0).repeat_interleave(B, dim=0).to(device)
        
#         neighbor_mask = (dist_XX <= self.neighbor_dist).type(torch.int) - torch.eye(T, device=device).unsqueeze(0).repeat(B, 1, 1)
#         non_neighbor_mask = (dist_XX > self.neighbor_dist).type(torch.int)
        
#         dist_z = torch.cdist(z_sample, z_sample)
#         neighbor_dist = ((dist_z * neighbor_mask).sum(dim=[1, 2]) / neighbor_mask.sum()).mean()
#         non_neighbor_dist = ((dist_z * non_neighbor_mask).sum(dim=[1, 2]) / non_neighbor_mask.sum()).mean()
#         tnc_loss = neighbor_dist - non_neighbor_dist
        
#         loss = recon_loss + kl_loss + self.tnc_reg * tnc_loss
#         loss.backward()
#         optimizer.step()
        
#         return {'loss': loss.item()}
    
#     def validation_step(self, o, **kwargs):
#         device = self.get_device()
#         o_recon, priors, posteriors = self(o)
        
#         prior_dist = Normal(*map(torch.stack, zip(*priors)))
#         posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#         kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
        
#         _, z_0 = self.get_init_state(o[..., 0], sample=False)
#         z_sample = torch.cat([z_0.unsqueeze(0), posterior_dist.rsample()], dim=0).transpose(0, 1)
        
#         B, T, *dims = z_sample.shape
#         dist_XX = torch.zeros(T, T)
#         for i_idx in range(T):
#             dist_XX[i_idx, :] = torch.abs(torch.arange(T) - i_idx)
#         dist_XX = dist_XX.unsqueeze(0).repeat_interleave(B, dim=0).to(device)
        
#         neighbor_mask = (dist_XX <= self.neighbor_dist).type(torch.int) - torch.eye(T, device=device).unsqueeze(0).repeat(B, 1, 1)
#         non_neighbor_mask = (dist_XX > self.neighbor_dist).type(torch.int)
        
#         dist_z = torch.cdist(z_sample, z_sample)
#         neighbor_dist = ((dist_z * neighbor_mask).sum(dim=[1, 2]) / neighbor_mask.sum()).mean()
#         non_neighbor_dist = ((dist_z * non_neighbor_mask).sum(dim=[1, 2]) / non_neighbor_mask.sum()).mean()
#         tnc_loss = neighbor_dist - non_neighbor_dist
        
#         loss = recon_loss + kl_loss + self.tnc_reg * tnc_loss
        
#         return {'loss': loss.item()}

#     def eval_step(self, dl, rollout_obs_num=1, **kwargs):
#         device = kwargs["device"]

#         recon_ = []
#         kl_ = []
#         dyn_error_ = []
#         tnc_ = []
        
#         for o, _ in dl: 
#             o = o.to(device)
#             B, C, W, H, T = o.shape
#             Z = self.lstate_size
            
#             # Mean Condition Number (Pullback)
#             o_ = o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H).to(device)
#             z, _ = self.state_posterior(o_)
#             G = get_pullbacked_Riemannian_metric(self.decoder, z, create_graph=False)
            
#             # Specific loss 
#             o_recon, priors, posteriors = self(o)

#             prior_dist = Normal(*map(torch.stack, zip(*priors)))
#             posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
#             kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

#             recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
            
#             z_sample = posterior_dist.rsample().view(-1, self.lstate_size)
            
#             recon_.append(recon_loss)
#             kl_.append(kl_loss)
            
#             # Rollout Error(Dynamics error)
#             lstates_rollout, o_rollout = self.rollout_multiple_observation(o=o[..., :rollout_obs_num], T_rollout=T-rollout_obs_num)
            
#             # GENERAL o_dynamics
#             o_dynamics = o[..., rollout_obs_num:]
            
#             dyn_error_.append(((o_rollout - o_dynamics)**2).sum((1, 2, 3, 4)).mean())  
            
#             _, z_0 = self.get_init_state(o[..., 0], sample=False)
#             z_sample = torch.cat([z_0.unsqueeze(0), posterior_dist.rsample()], dim=0).transpose(0, 1)
            
#             B, T, *dims = z_sample.shape
#             dist_XX = torch.zeros(T, T)
#             for i_idx in range(T):
#                 dist_XX[i_idx, :] = torch.abs(torch.arange(T) - i_idx)
#             dist_XX = dist_XX.unsqueeze(0).repeat_interleave(B, dim=0).to(device)
            
#             neighbor_mask = (dist_XX <= self.neighbor_dist).type(torch.int) - torch.eye(T, device=device).unsqueeze(0).repeat(B, 1, 1)
#             non_neighbor_mask = (dist_XX > self.neighbor_dist).type(torch.int)
            
#             dist_z = torch.cdist(z_sample, z_sample)
#             neighbor_dist = ((dist_z * neighbor_mask).sum(dim=[1, 2]) / neighbor_mask.sum()).mean()
#             non_neighbor_dist = ((dist_z * non_neighbor_mask).sum(dim=[1, 2]) / non_neighbor_mask.sum()).mean()
#             tnc_loss = neighbor_dist - non_neighbor_dist
                
#             tnc_.append(tnc_loss)
                
#         recon_ = torch.stack(recon_).mean()
#         kl_ = torch.stack(kl_).mean()
#         dyn_error_ = torch.stack(dyn_error_).mean()
#         tnc_ = torch.stack(tnc_).mean()
            
#         return {
#             "eval/recon_loss_": recon_.item(),
#             "eval/kl_loss_": kl_.item(),
#             "eval/dyn_error_": dyn_error_.item(),
#             "eval/tnc_loss_": tnc_.item(),
#         }

class RSSM_posterior_w_hidden(nn.Module):
    """Recurrent State Space Model for no-action, no-reward, only-dynamics setting, hstate connection for posterior
    """
    def __init__(self,
            encoder,
            decoder,
            hstate_dim=200,
            hidden_dim=200,
            activation='relu'
        ):
        super().__init__()
        self.hstate_dim = hstate_dim
        self.lstate_size = decoder.in_chan
        self.act_fn = getattr(F, activation)
        self.encoder = encoder
        self.decoder = decoder
        
        self.grucell = nn.GRUCell(self.hstate_dim, self.hstate_dim)
        self.lat_act_layer = nn.Linear(self.lstate_size, self.hstate_dim)
        
        self.fc_prior_1 = nn.Linear(self.hstate_dim, hidden_dim)
        self.fc_prior_m = nn.Linear(hidden_dim, self.lstate_size)
        self.fc_prior_s = nn.Linear(hidden_dim, self.lstate_size)
        
        self.fc_posterior_1 = nn.Linear(self.hstate_dim + hidden_dim, hidden_dim)
        self.fc_posterior_m = nn.Linear(hidden_dim, self.lstate_size)
        self.fc_posterior_s = nn.Linear(hidden_dim, self.lstate_size)

    def get_init_state(self, enc, h_t=None, s_t=None, sample=False):
        """Returns the initial posterior given the observation."""
        N, dev = enc.size(0), enc.device
        h_t = torch.zeros(N, self.hstate_dim).to(dev) if h_t is None else h_t
        s_t = torch.zeros(N, self.lstate_size).to(dev) if s_t is None else s_t
        h_tp1 = self.deterministic_state_fwd(h_t, s_t)
        if sample:
            s_tp1 = self.state_posterior(h_t, enc, sample=True)
        else:
            s_tp1, _ = self.state_posterior(h_t, enc)
        return h_tp1, s_tp1

    def deterministic_state_fwd(self, h_t, s_t):
        """Returns the deterministic state given the previous states
        and action.
        """
        h = self.act_fn(self.lat_act_layer(s_t))
        return self.grucell(h, h_t)

    def state_prior(self, h_t, sample=False):
        """Returns the state prior given the deterministic state."""
        z = self.act_fn(self.fc_prior_1(h_t))
        m = self.fc_prior_m(z)
        s = F.softplus(self.fc_prior_s(z)) + 1e-1
        if sample:
            return m + torch.randn_like(m) * s
        return m, s

    def state_posterior(self, h_t, o_t, sample=False):
        """Returns the state prior given the deterministic state and obs."""
        z_t = self.encoder(o_t)
        z_t = self.act_fn(self.fc_posterior_1(torch.cat([h_t, z_t], dim=-1)))
        m = self.fc_posterior_m(z_t)
        s = F.softplus(self.fc_posterior_s(z_t)) + 1e-1
        if sample:
            return m + torch.randn_like(m) * s
        return m, s

    def rollout_prior(self, o_t, T, h_t=None, s_t=None):
        h_t, s_t = self.get_init_state(o_t, h_t=h_t, s_t=s_t, sample=False)
        o_t_recon = self.decoder(s_t)
        
        lstates, o_recon = [s_t], [o_t_recon]
        
        for t_idx in range(1, T):
            h_tp1 = self.deterministic_state_fwd(h_t, s_t)
            s_tp1_pr_m, s_tp1_pr_s = self.state_prior(h_tp1)
            s_tp1_prior_dist = Normal(s_tp1_pr_m, s_tp1_pr_s)
            
            s_tp1 = s_tp1_prior_dist.rsample()
            o_tp1_recon = self.decoder(s_tp1)
            
            lstates.append(s_tp1)
            o_recon.append(o_tp1_recon)
            
            h_t, s_t = h_tp1, s_tp1
            
        return torch.stack(lstates, dim=-1), torch.stack(o_recon, dim=-1)
    
    def rollout_multiple_observation(self, o, T_rollout, h_t=None, s_t=None):
        """
        rollout with multiple frames of observation. rollout_prior can be replaced by this function,
        since single frame o_t will automatically skip the forward process because T_obs = 1.
        """ 
        
        ### forward on posteriors using observation (but no prior and recon are needed) ###
        """code from the forward function"""
        
        ## get the number of frames of observation
        if len(o.shape)==4:
            # (B, C, W, H)..single frame
            o = o.unsqueeze(-1)     # make it into (B, C, W, H, 1)
        elif len(o.shape)==5:
            # (B, C, W, H, T_obs).. multiple frames
            pass
        T_obs = o.shape[-1]         # number of frames of observation (becomes 1 if single frame, n if n frames)
        
        posteriors = []
        
        o_t = o[..., 0]     # first observation
        
        h_t, s_t = self.get_init_state(o_t, sample=False)       # h_1, s_1

        for t_idx in range(1, T_obs):

            h_tp1 = self.deterministic_state_fwd(h_t, s_t)

            o_tp1 = o[..., t_idx]
            s_tp1_po_m, s_tp1_po_s = self.state_posterior(h_t, o_tp1)
            s_tp1_posterior_dist = Normal(s_tp1_po_m, s_tp1_po_s)

            s_tp1 = s_tp1_posterior_dist.rsample()
            
            posteriors.append([s_tp1_po_m, s_tp1_po_s])

            h_t, s_t = h_tp1, s_tp1                             # In the last iteration, h_{T_obs}, s_{T_obs}
    
        lstates, o_recon = [], []
        
        # rollout starting from the final (h_t, s_t) obtained in forward process.
        for t_idx in range(T_rollout):
            h_tp1 = self.deterministic_state_fwd(h_t, s_t)
            s_tp1_pr_m, s_tp1_pr_s = self.state_prior(h_tp1)
            s_tp1_prior_dist = Normal(s_tp1_pr_m, s_tp1_pr_s)
            
            s_tp1 = s_tp1_prior_dist.rsample()
            o_tp1_recon = self.decoder(s_tp1)
            
            lstates.append(s_tp1)
            o_recon.append(o_tp1_recon)
            
            h_t, s_t = h_tp1, s_tp1
            
        return torch.stack(lstates, dim=-1), torch.stack(o_recon, dim=-1)

    def get_device(self):
        return list(self.parameters())[0].device
    
    def forward(self, o):
        B, C, W, H, T = o.shape
        
        o_recon, priors, posteriors = [], [], []
        
        # t=0
        o_t = o[..., 0]
        h_t, s_t = self.get_init_state(o_t, sample=False)

        o_t_recon = self.decoder(s_t)
        o_recon.append(o_t_recon)

        for t_idx in range(1, T):

            h_tp1 = self.deterministic_state_fwd(h_t, s_t)
            s_tp1_pr_m, s_tp1_pr_s = self.state_prior(h_tp1)
            s_tp1_prior_dist = Normal(s_tp1_pr_m, s_tp1_pr_s)

            o_tp1 = o[..., t_idx]
            s_tp1_po_m, s_tp1_po_s = self.state_posterior(h_t, o_tp1)
            s_tp1_posterior_dist = Normal(s_tp1_po_m, s_tp1_po_s)

            s_tp1 = s_tp1_posterior_dist.rsample()
            o_tp1_recon = self.decoder(s_tp1)

            o_recon.append(o_tp1_recon)
            priors.append([s_tp1_pr_m, s_tp1_pr_s])
            posteriors.append([s_tp1_po_m, s_tp1_po_s])

            h_t, s_t = h_tp1, s_tp1
            
        return o_recon, priors, posteriors
    
    def train_step(self, o, optimizer, **kwargs):
        
        device = self.get_device()
        optimizer.zero_grad()
        
        o_recon, priors, posteriors = self(o)

        prior_dist = Normal(*map(torch.stack, zip(*priors)))
        posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
        kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

        recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()

        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()
        
        return {'loss': loss.item()}
        # return {'loss': loss.item(), 'recon_loss': recon_loss.item(), 'kl_loss': kl_loss.item()}
        
    def validation_step(self, o, **kwargs):
        
        device = self.get_device()
        o_recon, priors, posteriors = self(o)
        
        prior_dist = Normal(*map(torch.stack, zip(*priors)))
        posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
        kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

        recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()

        loss = recon_loss + kl_loss
        
        return {'loss': loss.item()}
    
    def eval_step(self, dl, rollout_obs_num=1, **kwargs):
        device = self.get_device()
        
        score = []
        recon_ = []
        kl_ = []
        # dyn_error_latent = []
        # dyn_error_image = []
        dyn_error_ = []
        
        for o, _ in dl: 
            o = o.to(device)
            B, C, W, H, T = o.shape
            Z = self.lstate_size
            
            # Mean Condition Number
            _, _, posteriors = self(o)
            z = torch.cat([p[0] for p in posteriors], dim=0)
            
            # G = get_pullbacked_Riemannian_metric(self.decoder, z, create_graph=False)
            # score.append(get_flattening_scores(G, mode='condition_number'))
            
            # Specific loss 
            o_recon, priors, posteriors = self(o)

            prior_dist = Normal(*map(torch.stack, zip(*priors)))
            posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
            kl_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum((0, 2)), torch.ones(1).to(device)*3.0).mean()

            recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
            
            z_sample = posterior_dist.rsample().view(-1, self.lstate_size)
            
            recon_.append(recon_loss)
            kl_.append(kl_loss)
            
            # Rollout Error(Dynamics error)
            lstates_rollout, o_rollout = self.rollout_multiple_observation(o=o[..., :rollout_obs_num], T_rollout=T-rollout_obs_num)
            
            # GENERAL o_dynamics
            o_dynamics = o[..., rollout_obs_num:]
            
            dyn_error_.append(((o_rollout - o_dynamics)**2).sum((1, 2, 3, 4)).mean())      
            
        mean_condition_number = torch.cat(score).mean()
        recon_ = torch.stack(recon_).mean()
        kl_ = torch.stack(kl_).mean()
        dyn_error_ = torch.stack(dyn_error_).mean()
        
        return {
            "eval/MCN_": mean_condition_number.item(),
            "eval/recon_loss_": recon_.item(),
            "eval/kl_loss_": kl_.item(),
            "eval/dyn_error_": dyn_error_.item(),
        }
    
    def visualization_step(self, dl, rollout_obs_num=1, **kwargs):
        device = self.get_device()
        
        d_val = {}
        
        num_figures = 5
        x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]].to(self.get_device())
        
        lstates, o_recon = self.rollout_multiple_observation(x[..., :rollout_obs_num], dl.dataset.data.shape[-1])
        
        B, C, W, H, T = o_recon.shape
        
        img_grid = torch.cat([x[..., :rollout_obs_num], o_recon], dim=-1) # (B, C, W, H, T+rollout_obs_num)
        img_grid = img_grid.permute(0, 4, 1, 2, 3).reshape(B*(T+rollout_obs_num), C, W, H)
        img_grid = make_grid(img_grid.detach().cpu(), nrow=T+rollout_obs_num, value_range=(0, 1), pad_value=1)
        
        
        boundary = (W+2)*rollout_obs_num       # boundary line between observation and rollout
        img_grid[0,:,boundary:boundary+2] = 1
        img_grid[1,:,boundary:boundary+2] = 0
        img_grid[2,:,boundary:boundary+2] = 0
        
        d_val['rollout@'] = torch.clip(img_grid, min=0, max=1)
        
        # 2d graph (latent sapce)
        if self.lstate_size == 2:
            num_points_for_each_class = 200
            num_G_plots_for_each_class = 100
            num_rollout_for_each_class = 1
            num_rollout = 36

            label_unique = torch.unique(dl.dataset.targets)

            z_ = []
            z_sampled_ = []
            label_ = []
            label_sampled_ = []
            G_ = []
            z_rollout = []
            o_rollout = []
            label_rollout = []

            for label in label_unique:
                # temp_data = dl.dataset.data[dl.dataset.targets == label][:num_points_for_each_class][..., 0]
                # temp_z, _ = self.state_posterior(temp_data.to(self.get_device()))
                
                temp_data = dl.dataset.data[dl.dataset.targets == label][:num_points_for_each_class]
                _, _, temp_posterior = self(temp_data.to(device))
                temp_z = torch.cat([p[0] for p in temp_posterior], dim=0)
                
                z_sampled = temp_z[torch.randperm(len(temp_z))[:num_G_plots_for_each_class]]
                # G = get_pullbacked_Riemannian_metric(self.decoder, z_sampled)
                z_.append(temp_z)
                label_.append(label.repeat(temp_z.size(0)))
                z_sampled_.append(z_sampled)
                label_sampled_.append(label.repeat(z_sampled.size(0)))
                # G_.append(G)
                
                temp_data = dl.dataset.data[dl.dataset.targets == label][:num_rollout_for_each_class][..., :rollout_obs_num]
                lstates, o_recon = self.rollout_multiple_observation(temp_data.to(self.get_device()), num_rollout)
                z_rollout.append(lstates)
                o_rollout.append(o_recon)
                label_rollout.append(label.repeat(lstates.shape[0]))
                
            z_ = torch.cat(z_, dim=0).detach().cpu().numpy()
            label_ = torch.cat(label_, dim=0).detach().cpu().numpy()
            color_ = label_to_color(label_)
            G_ = torch.cat(G_, dim=0).detach().cpu()
            z_sampled_ = torch.cat(z_sampled_, dim=0).detach().cpu().numpy()
            label_sampled_ = torch.cat(label_sampled_, dim=0).detach().cpu().numpy()
            color_sampled_ = label_to_color(label_sampled_)
            z_rollout = torch.cat(z_rollout, dim=0).detach().cpu().numpy()
            o_rollout = torch.cat(o_rollout, dim=0).detach().cpu().numpy()
            label_rollout = torch.cat(label_rollout, dim=0).detach().cpu().numpy()
            color_rollout = label_to_color(label_rollout)
            
            f = plt.figure(figsize=(10, 10))
            plt.title('Latent space embeddings with equidistant ellipses')
            z_scale = np.minimum(np.max(z_, axis=0), np.min(z_, axis=0))
            eig_mean = torch.svd(G_).S.mean().item()
            scale = 0.1 * z_scale * np.sqrt(eig_mean)
            alpha = 0.3
            for idx in range(len(z_sampled_)):
                e = PD_metric_to_ellipse(np.linalg.inv(G_[idx,:,:]), z_sampled_[idx,:], scale, fc=color_sampled_[idx,:]/255.0, alpha=alpha)
                plt.gca().add_artist(e)
            for label in label_unique:
                label = label.item()
                plt.scatter(z_[label_==label,0], z_[label_==label,1], color=color_[label_==label][0]/255, label=label)
                
            for idx in range(len(z_rollout)):
                plt.plot(z_rollout[idx, 0, :], z_rollout[idx, 1, :], '-o', color='k')
                for t_idx in range(z_rollout.shape[-1]):
                    im = OffsetImage(o_rollout[idx, :, :, :, t_idx].transpose(1, 2, 0), zoom=0.7)
                    if t_idx < rollout_obs_num:             # given observations
                        ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=True)
                    elif t_idx == z_rollout.shape[-1]-1:     # the final frame of rollout
                        ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=True, bboxprops=dict(edgecolor='red'))    
                    else:
                        ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=False)
                    plt.gca().add_artist(ab)
            plt.legend()
            plt.axis('equal')
            plt.close()
            f_np = np.transpose(figure_to_array(f), (2, 0, 1))[:3,:,:]
            
            d_val['latent_space@'] = f_np
        
        
        # Step size plot of Dynamics Rollout
        num_rollout_for_each_class = 4
        num_rollout = 36

        label_unique = torch.unique(dl.dataset.targets)

        z_ = []
        z_sampled_ = []
        label_ = []
        label_sampled_ = []
        G_ = []
        z_rollout = []
        o_rollout = []
        label_rollout = []

        for label in label_unique:
            temp_data = dl.dataset.data[dl.dataset.targets == label][:num_rollout_for_each_class][..., :rollout_obs_num]
            lstates, o_recon = self.rollout_multiple_observation(temp_data.to(self.get_device()), num_rollout)
            z_rollout.append(lstates)
            o_rollout.append(o_recon)
            label_rollout.append(label.repeat(lstates.shape[0]))
            
        z_rollout = torch.cat(z_rollout, dim=0).detach().cpu()
        o_rollout = torch.cat(o_rollout, dim=0).detach().cpu()
        label_rollout = torch.cat(label_rollout, dim=0).detach().cpu()
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle('Stepsize of Dynamics Rollout', y=0.92, fontsize=20)
        for label in label_unique:
            z_ = z_rollout[label_rollout == label]
            o_ = o_rollout[label_rollout == label]
            for idx in range(4):
                ax_idx = (idx//2, idx%2)
                stepsize_ = (z_[idx, :, 1:] - z_[idx, :, :-1]).norm(dim=0)
                axs[ax_idx].plot(np.arange(len(stepsize_))+0.5, stepsize_, '-o', label=label.item())
                axs[ax_idx].legend()
        plt.close()
        f_np = np.transpose(figure_to_array(fig), (2, 0, 1))[:3,:,:]
        
        d_val['dyn_stepsize@'] = f_np
                
        return d_val

class RSSM_deterministic(nn.Module):
    """
    Recurrent State Space Model with deterministic functions (no variance term, only mean)
    Latent transition loss is MSE, not KL divergence
    """
    def __init__(self,
            encoder,
            decoder,
            hstate_dim=200,
            hidden_dim=200,
            activation='relu',
            **kwargs,
        ):
        super().__init__()
        self.hstate_dim = hstate_dim
        self.lstate_size = encoder.out_chan
        self.act_fn = getattr(F, activation)
        self.encoder = encoder
        self.decoder = decoder
        
        self.grucell = nn.GRUCell(self.hstate_dim, self.hstate_dim)
        self.lat_act_layer = nn.Linear(self.lstate_size, self.hstate_dim)
        
        self.fc_prior_1 = nn.Linear(self.hstate_dim, hidden_dim)
        self.fc_prior_m = nn.Linear(hidden_dim, self.lstate_size)
        
        self.rollout_obs_num = kwargs.get('rollout_obs_num', 1)
        print("init rollout_obs_num: ", self.rollout_obs_num)

    def get_init_state(self, o_t, h_t=None, s_t=None, sample=False):
        """Returns the initial posterior given the observation."""
        N, dev = o_t.size(0), o_t.device
        h_t = torch.zeros(N, self.hstate_dim).to(dev) if h_t is None else h_t
        s_t = torch.zeros(N, self.lstate_size).to(dev) if s_t is None else s_t
        # print(f"get_init_state, enc.shape={o_t.shape}")
        # print(f"get_init_state, h_t.shape={h_t.shape}")
        # print(f"get_init_state, s_t.shape={s_t.shape}")
        h_tp1 = self.deterministic_state_fwd(h_t, s_t)
        s_tp1 = self.state_posterior(o_t)
        # print(f"get_init_state, h_tp1.shape={h_tp1.shape}")
        # print(f"get_init_state, s_tp1.shape={s_tp1.shape}")
        return h_tp1, s_tp1

    def deterministic_state_fwd(self, h_t, s_t):
        """Returns the deterministic state given the previous states
        and action.
        """
        h = self.act_fn(self.lat_act_layer(s_t))
        # print(f"h shape: {h.shape}, h_t shape: {h_t.shape}")
        return self.grucell(h, h_t)

    def state_prior(self, h_t, sample=False):
        """Returns the state prior given the deterministic state."""
        z = self.act_fn(self.fc_prior_1(h_t))
        m = self.fc_prior_m(z)
        return m

    def state_posterior(self, o_t, sample=False):
        """Returns the state prior given the deterministic state and obs."""
        # print(f"state_posterior, o_t.shape={o_t.shape}")    # (10, *dims) as expected
        z = self.encoder(o_t)       # This should return (10,2), but got (7840,2)
        m = z[:, :self.lstate_size]
        # print(f"state_posterior, z.shape={z.shape}")
        # print(f"state_posterior, m.shape={m.shape}")
        return m

    def rollout_prior(self, o_t, T, h_t=None, s_t=None):
        h_t, s_t = self.get_init_state(o_t, h_t=h_t, s_t=s_t, sample=False)
        o_t_recon = self.decoder(s_t)
        
        lstates, o_recon = [s_t], [o_t_recon]
        
        for t_idx in range(1, T):
            h_tp1 = self.deterministic_state_fwd(h_t, s_t)
            s_tp1 = self.state_prior(h_tp1)
            o_tp1_recon = self.decoder(s_tp1)
            
            lstates.append(s_tp1)
            o_recon.append(o_tp1_recon)
            
            h_t, s_t = h_tp1, s_tp1
            
        return torch.stack(lstates, dim=-1), torch.stack(o_recon, dim=-1)
    
    def rollout_multiple_observation(self, o, T_rollout, h_t=None, s_t=None):
        ## get the number of frames of observation
        if len(o.shape)==4:
            o = o.unsqueeze(-1)     # make (B, C, W, H, 1)
        elif len(o.shape)==5:
            pass                    # already (B, C, W, H, 1)
        T_obs = o.shape[-1]         # number of frames of observation
        
        o_t = o[..., 0]     # first observation
        h_t, s_t = self.get_init_state(o_t, sample=False)       # h_1, s_1

        # forward with observation
        for t_idx in range(1, T_obs):
            h_tp1 = self.deterministic_state_fwd(h_t, s_t)
            o_tp1 = o[..., t_idx]
            s_tp1 = self.state_posterior(o_tp1)
            
            h_t, s_t = h_tp1, s_tp1
        
        # rollout from the final (h_t, s_t) obtained in forward.
        lstates, o_recon = [], []
        for t_idx in range(T_rollout):
            h_tp1 = self.deterministic_state_fwd(h_t, s_t)
            s_tp1= self.state_prior(h_tp1)
            o_tp1_recon = self.decoder(s_tp1)
            
            lstates.append(s_tp1)
            o_recon.append(o_tp1_recon)
            h_t, s_t = h_tp1, s_tp1
            
        return torch.stack(lstates, dim=-1), torch.stack(o_recon, dim=-1)

    def get_device(self):
        return list(self.parameters())[0].device
    
    def forward(self, o):
        B, C, W, H, T = o.shape
        
        o_recon, priors, posteriors = [], [], []
        
        # t=0
        o_t = o[..., 0]                             
        h_t, s_t = self.get_init_state(o_t, sample=False)

        o_t_recon = self.decoder(s_t)
        o_recon.append(o_t_recon)

        for t_idx in range(1, T):

            h_tp1 = self.deterministic_state_fwd(h_t, s_t)
            s_tp1_pr = self.state_prior(h_tp1)

            o_tp1 = o[..., t_idx]
            s_tp1_po  = self.state_posterior(o_tp1)
            o_tp1_recon = self.decoder(s_tp1_po)

            o_recon.append(o_tp1_recon)
            priors.append(s_tp1_pr)
            posteriors.append(s_tp1_po)

            h_t, s_t = h_tp1, s_tp1_po
            
        return o_recon, priors, posteriors
    
    def train_step(self, o, optimizer, **kwargs):
        """
        o: (B, *dims, T)    
        """        
        
        # print("RSSM_deterministic train_step, o.shape=", o.shape)        
        
        device = self.get_device()
        optimizer.zero_grad()
        
        # compute dynamics learning loss
        o_recon, priors, posteriors = self(o)
        latent_mse_loss = ((torch.stack(priors) - torch.stack(posteriors))**2).mean()
        recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()

        # compute and optimize total loss
        loss = recon_loss + latent_mse_loss
        loss.backward()
        optimizer.step()
        
        
        return {
            'loss': loss.item(),
            'loss/train_recon_loss_': recon_loss.item(),
            'loss/train_latent_mse_loss_': latent_mse_loss.item(),
        }   
        
    def validation_step(self, o, **kwargs):
        device = self.get_device()
        
        # compute dynamics learning loss
        o_recon, priors, posteriors = self(o)
        latent_mse_loss = ((torch.stack(priors) - torch.stack(posteriors))**2).mean()
        recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()

        # compute total loss
        loss = recon_loss + latent_mse_loss
        
        return {
            'loss': loss.item(),
            'loss/train_recon_loss_': recon_loss.item(),
            'loss/train_latent_mse_loss_': latent_mse_loss.item(),
        }  
    
    def eval_step(self, dl, rollout_obs_num=1, **kwargs):
        device = kwargs["device"]
        
        rollout_obs_num = self.rollout_obs_num
        print("eval step: rollout_obs_num=", rollout_obs_num)
        
        score = []
        recon_ = []
        kl_ = []
        # dyn_error_latent = []
        # dyn_error_image = []
        dyn_error_ = []
        
        for oyi in dl: 
            # oyi: list of o, y, idx
            o = oyi[0].to(device)
            # o = o.to(device)
            B, C, W, H, T = o.shape
            
            # Mean Condition Number
            o_ = o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H).to(device)
            z = self.state_posterior(o_)
            # G = get_pullbacked_Riemannian_metric(self.decoder, z, create_graph=False)
            # score.append(get_flattening_scores(G, mode='condition_number'))
            
            # Specific loss 
            o_recon, priors, posteriors = self(o)

            kl_loss = ((torch.stack(priors) - torch.stack(posteriors))**2).mean()
            recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
            
            recon_.append(recon_loss)
            kl_.append(kl_loss)
            
            # Rollout Error(Dynamics error)
            lstates_rollout, o_rollout = self.rollout_multiple_observation(o=o[..., :rollout_obs_num], T_rollout=T-rollout_obs_num)
            o_dynamics = o[..., rollout_obs_num:]
            
            dyn_error_.append(((o_rollout - o_dynamics)**2).sum((1, 2, 3, 4)).mean())      
            
        # mean_condition_number = torch.cat(score).mean()
        recon_ = torch.stack(recon_).mean()
        kl_ = torch.stack(kl_).mean()
        # dyn_error_latent = torch.stack(dyn_error_latent).mean()
        # dyn_error_image = torch.stack(dyn_error_image).mean()
        dyn_error_ = torch.stack(dyn_error_).mean()
        
        return {
            # "eval/MCN_": mean_condition_number.item(),
            "eval/recon_loss_": recon_.item(),
            "eval/kl_loss_": kl_.item(),
            # "eval/dyn_error_latent_": dyn_error_latent.item(),
            # "eval/dyn_error_image_": dyn_error_image.item(),
            "eval/dyn_error_": dyn_error_.item(),
        }
    
    def visualization_step(self, dl, rollout_obs_num=1, **kwargs):
        
        rollout_obs_num = self.rollout_obs_num
        print("vis step: rollout_obs_num=", rollout_obs_num)
        
        d_val = {}
        
        num_figures = 5
        x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]].to(self.get_device())
        
        lstates, o_recon = self.rollout_multiple_observation(x[..., :rollout_obs_num], dl.dataset.data.shape[-1]-rollout_obs_num)
        
        B, C, W, H, T = o_recon.shape
        
        img_grid = torch.cat([x[..., :rollout_obs_num], o_recon], dim=-1) # (B, C, W, H, T+rollout_obs_num)
        img_grid = img_grid.permute(0, 4, 1, 2, 3).reshape(B*(T+rollout_obs_num), C, W, H)
        img_grid = make_grid(img_grid.detach().cpu(), nrow=T+rollout_obs_num, value_range=(0, 1), pad_value=1)
        
        boundary = (W+2)*rollout_obs_num       # boundary line between observation and rollout
        img_grid[0,:,boundary:boundary+2] = 1
        img_grid[1,:,boundary:boundary+2] = 0
        img_grid[2,:,boundary:boundary+2] = 0
        
        d_val['rollout@'] = torch.clip(img_grid, min=0, max=1)
        
        # 2d graph (latent sapce)
        if self.lstate_size == 2:
            num_points_for_each_class = 200
            num_G_plots_for_each_class = 100
            num_rollout_for_each_class = 1
            num_rollout = dl.dataset.data.shape[-1]-rollout_obs_num

            label_unique = torch.unique(dl.dataset.targets)

            z_ = []
            z_sampled_ = []
            label_ = []
            label_sampled_ = []
            # G_ = []
            z_rollout = []
            o_rollout = []
            label_rollout = []

            for label in label_unique:
                temp_data = dl.dataset.data[dl.dataset.targets == label][:num_points_for_each_class][..., 0]
                temp_z = self.state_posterior(temp_data.to(self.get_device()))
                
                z_sampled = temp_z[torch.randperm(len(temp_z))[:num_G_plots_for_each_class]]
                # G = get_pullbacked_Riemannian_metric(self.decoder, z_sampled)
                z_.append(temp_z)
                label_.append(label.repeat(temp_z.size(0)))
                z_sampled_.append(z_sampled)
                label_sampled_.append(label.repeat(z_sampled.size(0)))
                # G_.append(G)
                
                temp_data = dl.dataset.data[dl.dataset.targets == label][:num_rollout_for_each_class][..., :rollout_obs_num]
                lstates, o_recon = self.rollout_multiple_observation(temp_data.to(self.get_device()), num_rollout)
                z_rollout.append(lstates)
                o_rollout.append(o_recon)
                label_rollout.append(label.repeat(lstates.shape[0]))
                
            z_ = torch.cat(z_, dim=0).detach().cpu().numpy()
            label_ = torch.cat(label_, dim=0).detach().cpu().numpy()
            color_ = label_to_color(label_)
            # G_ = torch.cat(G_, dim=0).detach().cpu()
            z_sampled_ = torch.cat(z_sampled_, dim=0).detach().cpu().numpy()
            label_sampled_ = torch.cat(label_sampled_, dim=0).detach().cpu().numpy()
            # color_sampled_ = label_to_color(label_sampled_)
            z_rollout = torch.cat(z_rollout, dim=0).detach().cpu().numpy()
            o_rollout = torch.cat(o_rollout, dim=0).detach().cpu().numpy()
            label_rollout = torch.cat(label_rollout, dim=0).detach().cpu().numpy()
            # color_rollout = label_to_color(label_rollout)
            
            f = plt.figure(figsize=(10, 10))
            plt.title('Latent space embeddings with equidistant ellipses')
            # z_scale = np.minimum(np.max(z_, axis=0), np.min(z_, axis=0))
            # eig_mean = torch.svd(G_).S.mean().item()
            # scale = 0.1 * z_scale * np.sqrt(eig_mean)
            # alpha = 0.3
            # for idx in range(len(z_sampled_)):
            #     e = PD_metric_to_ellipse(np.linalg.inv(G_[idx,:,:]), z_sampled_[idx,:], scale, fc=color_sampled_[idx,:]/255.0, alpha=alpha)
            #     plt.gca().add_artist(e)
            for label in label_unique:
                label = label.item()
                plt.scatter(z_[label_==label,0], z_[label_==label,1], color=color_[label_==label][0]/255, label=label)
                
            for idx in range(len(z_rollout)):
                plt.plot(z_rollout[idx, 0, :], z_rollout[idx, 1, :], '-o', color='k')
                for t_idx in range(z_rollout.shape[-1]):
                    im = OffsetImage(o_rollout[idx, :, :, :, t_idx].transpose(1, 2, 0), zoom=0.7)
                    if t_idx < rollout_obs_num:             # given observations
                        ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=True)
                    elif t_idx == z_rollout.shape[-1]-1:     # the final frame of rollout
                        ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=True, bboxprops=dict(edgecolor='red'))    
                    else:
                        ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=False)
                    plt.gca().add_artist(ab)
            plt.legend()
            plt.axis('equal')
            plt.close()
            f_np = np.transpose(figure_to_array(f), (2, 0, 1))[:3,:,:]
            
            d_val['latent_space@'] = f_np
        
        # Step size plot of Dynamics Rollout
        num_rollout_for_each_class = 4
        num_rollout = dl.dataset.data.shape[-1]-rollout_obs_num

        label_unique = torch.unique(dl.dataset.targets)

        z_ = []
        z_sampled_ = []
        label_ = []
        label_sampled_ = []
        G_ = []
        z_rollout = []
        o_rollout = []
        label_rollout = []

        for label in label_unique:
            temp_data = dl.dataset.data[dl.dataset.targets == label][:num_rollout_for_each_class][..., :rollout_obs_num]
            lstates, o_recon = self.rollout_multiple_observation(temp_data.to(self.get_device()), num_rollout)
            z_rollout.append(lstates)
            o_rollout.append(o_recon)
            label_rollout.append(label.repeat(lstates.shape[0]))
            
        z_rollout = torch.cat(z_rollout, dim=0).detach().cpu()
        o_rollout = torch.cat(o_rollout, dim=0).detach().cpu()
        label_rollout = torch.cat(label_rollout, dim=0).detach().cpu()
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle('Stepsize of Dynamics Rollout', y=0.92, fontsize=20)
        for label in label_unique:
            z_ = z_rollout[label_rollout == label]
            o_ = o_rollout[label_rollout == label]
            for idx in range(4):
                ax_idx = (idx//2, idx%2)
                stepsize_ = (z_[idx, :, 1:] - z_[idx, :, :-1]).norm(dim=0) / np.sqrt(self.lstate_size)
                axs[ax_idx].plot(np.arange(len(stepsize_))+0.5, stepsize_, '-o', label=label.item())
                axs[ax_idx].legend()
        plt.close()
        f_np = np.transpose(figure_to_array(fig), (2, 0, 1))[:3,:,:]
        
        d_val['dyn_stepsize@'] = f_np
        
        return d_val

# class ERRSSM_deterministic(RSSM_deterministic):
#     """
#     Encoder-regularized Recurrent State Space Model with deterministic functions (no variance term, only mean)
#     Latent transition loss is MSE, not KL divergence
#     Iso loss for the encoder is included...
#     """
#     def __init__(self,
#             encoder,
#             decoder,
#             hstate_dim=200,
#             hidden_dim=200,
#             activation='relu',
#             iso_reg=1.0,
#             bdry_reg=0.0,
#             measure='irvae',
#             bdry_measure='time_bdry',
#             distfunc = 'TimeSteps',
#             bandwidth = 2
#         ):
#         super().__init__(encoder=encoder, 
#                             decoder=decoder, 
#                             hstate_dim=hstate_dim, 
#                             hidden_dim=hidden_dim, 
#                             activation=activation)
        
#         self.iso_reg = iso_reg
#         self.bdry_reg = bdry_reg
#         self.measure = measure
#         self.bdry_measure = bdry_measure
#         self.distfunc = distfunc
#         self.bandwidth = bandwidth
        
#         self.c = 1/4
    
#     def train_step(self, o, optimizer, **kwargs):
#         """
#         o: (B, *dims, T)
#         o_L: (B, T, *dims)    
#         """        
#         device = self.get_device()
#         optimizer.zero_grad()
        
#         # compute dynamics learning loss
#         o_recon, priors, posteriors = self(o)
#         latent_mse_loss = ((torch.stack(priors) - torch.stack(posteriors))**2).mean()
#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
        
#         # compute embedding (to compute pushforward metric)
#         z_0 = self.state_posterior(o[..., 0], sample=False).unsqueeze(0) # (1, B, latent_dim)
#         for i in range(len(posteriors)):
#             posteriors[i] = posteriors[i].unsqueeze(0)
#         posterior_tensor = torch.cat(posteriors, dim=0)                 # (T-1, B, latent_dim)
#         z = torch.cat([z_0, posterior_tensor], dim=0).transpose(0,1)    # (B, T, latent_dim)
        
#         # compute iso loss and boundary loss
#         o_L = o.unsqueeze(1).transpose(1,-1).flatten(2,-1)      # (B, T, *dims)
#         L = get_laplacian(o_L, torch.zeros(o_L.shape[:2]), distfunc='TimeSteps', bandwidth=self.bandwidth)       # (B, T, T)
#         H_tilde = get_JGinvJT(L, z)
#         iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde, eta=0.2, measure=self.measure)
#         bdry_loss = boundary_loss(o_L, z, bdry_measure=self.bdry_measure)
        
#         # compute and optimize total loss
#         if self.measure == 'irvae':
#             loss = (recon_loss + latent_mse_loss) + self.iso_reg * iso_loss
#         elif self.measure == 'harmonic':
#             loss = (recon_loss + latent_mse_loss) - self.iso_reg * self.bdry_reg * bdry_loss
#         else:
#             raise NotImplementedError
#         loss.backward()
#         optimizer.step()
        
#         return {
#             'loss': loss.item(),
#             'loss/train_recon_loss_': recon_loss.item(),
#             'loss/train_latent_mse_loss_': latent_mse_loss.item(),
#             'loss/train_iso_loss_': iso_loss.item(),
#         } 
        
#     def validation_step(self, o, **kwargs):
#         device = self.get_device()
        
#         # compute dynamics learning loss
#         o_recon, priors, posteriors = self(o)
#         latent_mse_loss = ((torch.stack(priors) - torch.stack(posteriors))**2).mean()
#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
        
#         # compute embedding (to compute pushforward metric)
#         z_0 = self.state_posterior(o[..., 0], sample=False).unsqueeze(0) # (1, B, latent_dim)
#         for i in range(len(posteriors)):
#             posteriors[i] = posteriors[i].unsqueeze(0)
#         posterior_tensor = torch.cat(posteriors, dim=0)                 # (T-1, B, latent_dim)
#         z = torch.cat([z_0, posterior_tensor], dim=0).transpose(0,1)    # (B, T, latent_dim)
        
#         # compute iso loss and boundary loss
#         o_L = o.unsqueeze(1).transpose(1,-1).flatten(2,-1)      # (B, T, *dims)
#         L = get_laplacian(o_L, torch.zeros(o_L.shape[:2]), distfunc='TimeSteps', bandwidth=self.bandwidth)       # (B, T, T)
#         H_tilde = get_JGinvJT(L, z)
#         iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde, eta=0.2, measure=self.measure)
#         bdry_loss = boundary_loss(o_L, z, bdry_measure=self.bdry_measure)
        
#         # compute total loss
#         if self.measure == 'irvae':
#             loss = (recon_loss + latent_mse_loss) + self.iso_reg * iso_loss
#         elif self.measure == 'harmonic':
#             loss = (recon_loss + latent_mse_loss) - self.iso_reg * self.bdry_reg * bdry_loss
#         else:
#             raise NotImplementedError
        
#         return {
#             'loss': loss.item(),
#             'loss/train_recon_loss_': recon_loss.item(),
#             'loss/train_latent_mse_loss_': latent_mse_loss.item(),
#             'loss/train_iso_loss_': iso_loss.item(),
#         }
    
#     def eval_step(self, dl, rollout_obs_num=1, **kwargs):
#         device = kwargs["device"]
        
#         score = []
#         recon_ = []
#         kl_ = []
#         # dyn_error_latent = []
#         # dyn_error_image = []
#         dyn_error_ = []
        
#         for o, _ in dl: 
#             o = o.to(device)
#             B, C, W, H, T = o.shape
            
#             # Mean Condition Number
#             o_ = o.permute(0, 4, 1, 2, 3).reshape(B*T, C, W, H).to(device)
#             z = self.state_posterior(o_)
#             G = get_pullbacked_Riemannian_metric(self.decoder, z, create_graph=False)
#             score.append(get_flattening_scores(G, mode='condition_number'))
            
#             # Specific loss 
#             o_recon, priors, posteriors = self(o)

#             kl_loss = ((torch.stack(priors) - torch.stack(posteriors))**2).mean()
#             recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
            
#             recon_.append(recon_loss)
#             kl_.append(kl_loss)
            
#             # Rollout Error(Dynamics error)
#             lstates_rollout, o_rollout = self.rollout_multiple_observation(o=o[..., :rollout_obs_num], T_rollout=T-rollout_obs_num)
#             o_dynamics = o[..., rollout_obs_num:]
            
#             dyn_error_.append(((o_rollout - o_dynamics)**2).sum((1, 2, 3, 4)).mean())      
            
#         mean_condition_number = torch.cat(score).mean()
#         recon_ = torch.stack(recon_).mean()
#         kl_ = torch.stack(kl_).mean()
#         # dyn_error_latent = torch.stack(dyn_error_latent).mean()
#         # dyn_error_image = torch.stack(dyn_error_image).mean()
#         dyn_error_ = torch.stack(dyn_error_).mean()
        
#         return {
#             "eval/MCN_": mean_condition_number.item(),
#             "eval/recon_loss_": recon_.item(),
#             "eval/kl_loss_": kl_.item(),
#             # "eval/dyn_error_latent_": dyn_error_latent.item(),
#             # "eval/dyn_error_image_": dyn_error_image.item(),
#             "eval/dyn_error_": dyn_error_.item(),
#         }
    
#     def visualization_step(self, dl, rollout_obs_num=1, **kwargs):
        
#         d_val = {}
        
#         num_figures = 5
#         x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]].to(self.get_device())
        
#         lstates, o_recon = self.rollout_multiple_observation(x[..., :rollout_obs_num], dl.dataset.data.shape[-1])
        
#         B, C, W, H, T = o_recon.shape
        
#         img_grid = torch.cat([x[..., :rollout_obs_num], o_recon], dim=-1) # (B, C, W, H, T+rollout_obs_num)
#         img_grid = img_grid.permute(0, 4, 1, 2, 3).reshape(B*(T+rollout_obs_num), C, W, H)
#         img_grid = make_grid(img_grid.detach().cpu(), nrow=T+rollout_obs_num, value_range=(0, 1), pad_value=1)
        
#         boundary = (W+2)*rollout_obs_num       # boundary line between observation and rollout
#         img_grid[0,:,boundary:boundary+2] = 1
#         img_grid[1,:,boundary:boundary+2] = 0
#         img_grid[2,:,boundary:boundary+2] = 0
        
#         d_val['rollout@'] = torch.clip(img_grid, min=0, max=1)
        
#         # 2d graph (latent sapce)
#         if self.lstate_size == 2:
#             num_points_for_each_class = 200
#             num_G_plots_for_each_class = 100
#             num_rollout_for_each_class = 1
#             num_rollout = 36

#             label_unique = torch.unique(dl.dataset.targets)

#             z_ = []
#             z_sampled_ = []
#             label_ = []
#             label_sampled_ = []
#             G_ = []
#             z_rollout = []
#             o_rollout = []
#             label_rollout = []

#             for label in label_unique:
#                 temp_data = dl.dataset.data[dl.dataset.targets == label][:num_points_for_each_class][..., 0]
#                 temp_z = self.state_posterior(temp_data.to(self.get_device()))
                
#                 z_sampled = temp_z[torch.randperm(len(temp_z))[:num_G_plots_for_each_class]]
#                 G = get_pullbacked_Riemannian_metric(self.decoder, z_sampled)
#                 z_.append(temp_z)
#                 label_.append(label.repeat(temp_z.size(0)))
#                 z_sampled_.append(z_sampled)
#                 label_sampled_.append(label.repeat(z_sampled.size(0)))
#                 G_.append(G)
                
#                 temp_data = dl.dataset.data[dl.dataset.targets == label][:num_rollout_for_each_class][..., :rollout_obs_num]
#                 lstates, o_recon = self.rollout_multiple_observation(temp_data.to(self.get_device()), num_rollout)
#                 z_rollout.append(lstates)
#                 o_rollout.append(o_recon)
#                 label_rollout.append(label.repeat(lstates.shape[0]))
                
#             z_ = torch.cat(z_, dim=0).detach().cpu().numpy()
#             label_ = torch.cat(label_, dim=0).detach().cpu().numpy()
#             color_ = label_to_color(label_)
#             G_ = torch.cat(G_, dim=0).detach().cpu()
#             z_sampled_ = torch.cat(z_sampled_, dim=0).detach().cpu().numpy()
#             label_sampled_ = torch.cat(label_sampled_, dim=0).detach().cpu().numpy()
#             color_sampled_ = label_to_color(label_sampled_)
#             z_rollout = torch.cat(z_rollout, dim=0).detach().cpu().numpy()
#             o_rollout = torch.cat(o_rollout, dim=0).detach().cpu().numpy()
#             label_rollout = torch.cat(label_rollout, dim=0).detach().cpu().numpy()
#             color_rollout = label_to_color(label_rollout)
            
#             f = plt.figure(figsize=(10, 10))
#             plt.title('Latent space embeddings with equidistant ellipses')
#             z_scale = np.minimum(np.max(z_, axis=0), np.min(z_, axis=0))
#             eig_mean = torch.svd(G_).S.mean().item()
#             scale = 0.1 * z_scale * np.sqrt(eig_mean)
#             alpha = 0.3
#             for idx in range(len(z_sampled_)):
#                 e = PD_metric_to_ellipse(np.linalg.inv(G_[idx,:,:]), z_sampled_[idx,:], scale, fc=color_sampled_[idx,:]/255.0, alpha=alpha)
#                 plt.gca().add_artist(e)
#             for label in label_unique:
#                 label = label.item()
#                 plt.scatter(z_[label_==label,0], z_[label_==label,1], color=color_[label_==label][0]/255, label=label)
                
#             for idx in range(len(z_rollout)):
#                 plt.plot(z_rollout[idx, 0, :], z_rollout[idx, 1, :], '-o', color='k')
#                 for t_idx in range(z_rollout.shape[-1]):
#                     im = OffsetImage(o_rollout[idx, :, :, :, t_idx].transpose(1, 2, 0), zoom=0.7)
#                     if t_idx < rollout_obs_num:             # given observations
#                         ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=True)
#                     elif t_idx == z_rollout.shape[-1]-1:     # the final frame of rollout
#                         ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=True, bboxprops=dict(edgecolor='red'))    
#                     else:
#                         ab = AnnotationBbox(im, (z_rollout[idx, 0, t_idx], z_rollout[idx, 1, t_idx]), xycoords='data', frameon=False)
#                     plt.gca().add_artist(ab)
#             plt.legend()
#             plt.axis('equal')
#             plt.close()
#             f_np = np.transpose(figure_to_array(f), (2, 0, 1))[:3,:,:]
            
#             d_val['latent_space@'] = f_np
        
#         # Step size plot of Dynamics Rollout
#         num_rollout_for_each_class = 4
#         num_rollout = 36

#         label_unique = torch.unique(dl.dataset.targets)

#         z_ = []
#         z_sampled_ = []
#         label_ = []
#         label_sampled_ = []
#         G_ = []
#         z_rollout = []
#         o_rollout = []
#         label_rollout = []

#         for label in label_unique:
#             temp_data = dl.dataset.data[dl.dataset.targets == label][:num_rollout_for_each_class][..., :rollout_obs_num]
#             lstates, o_recon = self.rollout_multiple_observation(temp_data.to(self.get_device()), num_rollout)
#             z_rollout.append(lstates)
#             o_rollout.append(o_recon)
#             label_rollout.append(label.repeat(lstates.shape[0]))
            
#         z_rollout = torch.cat(z_rollout, dim=0).detach().cpu()
#         o_rollout = torch.cat(o_rollout, dim=0).detach().cpu()
#         label_rollout = torch.cat(label_rollout, dim=0).detach().cpu()
        
#         fig, axs = plt.subplots(2, 2, figsize=(16, 8))
#         fig.suptitle('Stepsize of Dynamics Rollout', y=0.92, fontsize=20)
#         for label in label_unique:
#             z_ = z_rollout[label_rollout == label]
#             o_ = o_rollout[label_rollout == label]
#             for idx in range(4):
#                 ax_idx = (idx//2, idx%2)
#                 stepsize_ = (z_[idx, :, 1:] - z_[idx, :, :-1]).norm(dim=0) / np.sqrt(self.lstate_size)
#                 axs[ax_idx].plot(np.arange(len(stepsize_))+0.5, stepsize_, '-o', label=label.item())
#                 axs[ax_idx].legend()
#         plt.close()
#         f_np = np.transpose(figure_to_array(fig), (2, 0, 1))[:3,:,:]
        
#         d_val['dyn_stepsize@'] = f_np
        
#         return d_val

# class ERRSSM_RobotArm_deterministic(RSSM_deterministic):
#     def __init__(self,
#             encoder,
#             decoder,
#             hstate_dim=200,
#             hidden_dim=200,
#             activation='relu',
#             iso_reg=1.0,
#             measure='irvae',
#             bandwidth=1
#         ):
#         super().__init__(
#             encoder=encoder, 
#             decoder=decoder, 
#             hstate_dim=hstate_dim, 
#             hidden_dim=hidden_dim, 
#             activation=activation
#         )
        
#         self.iso_reg = iso_reg
#         self.measure = measure
#         self.bandwidth = bandwidth
        
#     def train_step(self, o, y, optimizer, **kwargs):
#         """
#         o: (B, *dims, T)
#         y: latent vector (x, y coord. of cube), (B, 2, T)
#         """
#         device = self.get_device()
#         optimizer.zero_grad()
        
#         # compute dynamics learning loss
#         o_recon, priors, posteriors = self(o)
#         latent_mse_loss = ((torch.stack(priors) - torch.stack(posteriors))**2).mean()
#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
#         # recon_loss = torch.nn.MSELoss()(o, torch.stack(o_recon, dim=-1))
        
#         # compute embedding (to compute pushforward metric)
#         _, z_0 = self.get_init_state(o[..., 0], sample=False)
#         # posterior_mean = torch.stack([m for [m,s] in posteriors], dim=0)
#         posterior_mean = torch.stack(posteriors, dim=0)
#         z = torch.cat([z_0.unsqueeze(0), posterior_mean], dim=0).transpose(0,1) # mean instead of sample, (B, T, zdim)
#         # z_sample = torch.cat([z_0.unsqueeze(0), posterior_dist.rsample()], dim=0).transpose(0, 1)
        
#         z = z.flatten(0, 1) # (B*T, zdim)
#         y = y.transpose(1, 2).flatten(0, 1) # (B*T, 2)
        
#         # compute iso loss and boundary loss
#         L = get_laplacian(y.unsqueeze(0), torch.zeros(1, y.shape[0]), distfunc='Euclidean', bandwidth=self.bandwidth) # (1, B*T, B*T)
#         H_tilde = get_JGinvJT(L, z.unsqueeze(0)) # (1, B*T, zdim, zdim)
#         iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde, eta=0.2, measure=self.measure)
        
#         # compute and optimize total loss
#         loss = recon_loss + latent_mse_loss + self.iso_reg * iso_loss
#         loss.backward()
#         optimizer.step()

#         return {
#             'loss': loss.item(),
#             'loss/train_recon_loss_': recon_loss.item(),
#             'loss/train_latent_mse_loss_': latent_mse_loss.item(),
#             'loss/train_iso_loss_': iso_loss.item(),
#         }
    
#     def validation_step(self, o, y, **kwargs):
#         device = self.get_device()
        
#         # compute dynamics learning loss
#         o_recon, priors, posteriors = self(o)
#         latent_mse_loss = ((torch.stack(priors) - torch.stack(posteriors))**2).mean()
#         recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
#         # recon_loss = torch.nn.MSELoss()(o, torch.stack(o_recon, dim=-1))
        
#         # compute embedding (to compute pushforward metric)
#         _, z_0 = self.get_init_state(o[..., 0], sample=False)
#         # posterior_mean = torch.stack([m for [m,s] in posteriors], dim=0)
#         posterior_mean = torch.stack(posteriors, dim=0)
#         z = torch.cat([z_0.unsqueeze(0), posterior_mean], dim=0).transpose(0,1) # mean instead of sample, (B, T, zdim)
#         # z_sample = torch.cat([z_0.unsqueeze(0), posterior_dist.rsample()], dim=0).transpose(0, 1)
        
#         z = z.flatten(0, 1) # (B*T, zdim)
#         y = y.transpose(1, 2).flatten(0, 1) # (B*T, 2)
        
#         # compute iso loss and boundary loss
#         L = get_laplacian(y.unsqueeze(0), torch.zeros(1, y.shape[0]), distfunc='Euclidean', bandwidth=self.bandwidth) # (1, B*T, B*T)
#         H_tilde = get_JGinvJT(L, z.unsqueeze(0)) # (1, B*T, zdim, zdim)
#         iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde, eta=0.2, measure=self.measure)
        
#         # compute and optimize total loss
#         loss = recon_loss + latent_mse_loss + self.iso_reg * iso_loss
        
#         return {
#             'loss': loss.item(),
#             'loss/valid_recon_loss_': recon_loss.item(),
#             'loss/valid_latent_mse_loss_': latent_mse_loss.item(),
#             'loss/valid_iso_loss_': iso_loss.item(),
#         }
    
#     def eval_step(self, dl, rollout_obs_num=1, **kwargs):
#         device = kwargs["device"]
        
#         recon_ = []
#         latent_mse_ = []
#         dyn_error_ = []
        
#         for o, _ in dl: 
#             o = o.to(device)
#             B, C, W, H, T = o.shape
#             Z = self.lstate_size
            
#             # Specific loss 
#             o_recon, priors, posteriors = self(o)
#             latent_mse_loss = ((torch.stack(priors) - torch.stack(posteriors))**2).mean()
#             recon_loss = torch.nn.functional.mse_loss(o, torch.stack(o_recon, dim=-1), reduction='none').sum((1, 2, 3, 4)).mean()
#             # recon_loss = torch.nn.MSELoss()(o, torch.stack(o_recon, dim=-1))
            
#             recon_.append(recon_loss)
#             latent_mse_.append(latent_mse_loss)
            
#             # Rollout Error(Dynamics error)
#             lstates_rollout, o_rollout = self.rollout_multiple_observation(o=o[..., :rollout_obs_num], T_rollout=T-rollout_obs_num)
            
#             # GENERAL o_dynamics
#             o_dynamics = o[..., rollout_obs_num:]
            
#             dyn_error_.append(((o_rollout - o_dynamics)**2).sum((1, 2, 3, 4)).mean())  
#             # dyn_error_.append(torch.nn.MSELoss()(o_rollout, o_dynamics))
            
#         recon_ = torch.stack(recon_).mean()
#         latent_mse_ = torch.stack(latent_mse_).mean()
#         dyn_error_ = torch.stack(dyn_error_).mean()
        
#         return {
#             "eval/recon_loss_": recon_.item(),
#             "eval/latent_mse_loss_": latent_mse_.item(),
#             "eval/dyn_error_": dyn_error_.item(),
#         }
        
#     def visualization_step(self, dl, rollout_obs_num=1, **kwargs):
#         d_vis = {}
        
#         num_figures = 5
#         x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]].to(self.get_device())
        
#         lstates, o_recon = self.rollout_multiple_observation(x[..., :rollout_obs_num], dl.dataset.data.shape[-1])
        
#         B, C, W, H, T = o_recon.shape
        
#         img_grid = torch.cat([x[..., :rollout_obs_num], o_recon], dim=-1) # (B, C, W, H, T+rollout_obs_num)
#         img_grid = img_grid.permute(0, 4, 1, 2, 3).reshape(B*(T+rollout_obs_num), C, W, H)
#         img_grid = make_grid(img_grid.detach().cpu(), nrow=T+rollout_obs_num, value_range=(0, 1), pad_value=1)
        
#         boundary = (W+2)*rollout_obs_num       # boundary line between observation and rollout
#         img_grid[0,:,boundary:boundary+2] = 1
#         img_grid[1,:,boundary:boundary+2] = 0
#         img_grid[2,:,boundary:boundary+2] = 0
        
#         d_vis['rollout@'] = torch.clip(img_grid, min=0, max=1)
        
#         n_plot_series = 10
#         for x, y in dl:
#             x = x.to(self.get_device())[:n_plot_series]
#             y = y.to(self.get_device())[:n_plot_series]
#             break
        
#         o_recon, priors, posteriors = self(x)

#         _, z_0 = self.get_init_state(x[..., 0], sample=False)
#         posterior_mean = torch.stack(posteriors, dim=0)
#         z = torch.cat([z_0.unsqueeze(0), posterior_mean], dim=0).transpose(0,1) # mean instead of sample

#         z_rollout, recon_rollout = self.rollout_multiple_observation(x[..., :rollout_obs_num], T_rollout=x.shape[-1])

#         plotly_layout = dict(margin=dict(l=20, r=20, t=20, b=20))
#         fig = make_subplots(rows=1, cols=3, subplot_titles=['z', 'z_rollout', 'y'])
#         fig.update_layout(**plotly_layout, width=1600, height=500)
#         colors = px.colors.qualitative.Dark24
#         for b_idx in range(len(z)):
#             fig.add_trace(
#                 go.Scatter(
#                     x=z[b_idx, :, 0].detach().cpu(), 
#                     y=z[b_idx, :, 1].detach().cpu(), 
#                     mode='lines+markers', 
#                     marker=dict(size=10, color=colors[b_idx%len(colors)]), 
#                     line_color=colors[b_idx%len(colors)], showlegend=False
#                 ), row=1, col=1
#             )
            
#             fig.add_trace(
#                 go.Scatter(
#                     x=y[b_idx, 0, :].detach().cpu(), 
#                     y=y[b_idx, 1, :].detach().cpu(), 
#                     mode='lines+markers', 
#                     marker=dict(size=10, color=colors[b_idx%len(colors)]), 
#                     line_color=colors[b_idx%len(colors)], showlegend=False
#                 ), row=1, col=3
#             )
            
#             fig.add_trace(
#                 go.Scatter(
#                     x=z_rollout[b_idx, 0, :].detach().cpu(), 
#                     y=z_rollout[b_idx, 1, :].detach().cpu(), 
#                     mode='lines+markers', 
#                     marker=dict(size=10, color=colors[b_idx%len(colors)]), 
#                     line_color=colors[b_idx%len(colors)], showlegend=False
#                 ), row=1, col=2
#             )
            
#         fig_img = plotly_fig2array(fig)
#         fig_img = np.transpose(fig_img, (2, 0, 1))[:3,:,:]
#         d_vis['latent@'] = fig_img
        
#         return d_vis
    
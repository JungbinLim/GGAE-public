#############################################
#                                           #
# code from IRVAE_public (Lee et al., 2022) #
#                                           #
#############################################

import os
from omegaconf import OmegaConf
import torch

from models.ae import (
    AE,
    GGAE,
    TIMESERIES_AE,
    TIMESERIES_GGAE,
)

from models.modules import (
    FC_vec,
    FC_image,
    # IsotropicGaussian,
    ConvNet28,
    DeConvNet28,
    ConvNet64,
    DeConvNet64
)

from models.rssm import (
    RecurrentStateSpaceModel,
    # IsoRSSM,
    # NRRSSM,
    # RSSM_posterior_w_hidden,
    # ERRSSM,
    RSSM_deterministic,
    # ERRSSM_deterministic,
    # TNCRSSM,
    # ERRSSM_RobotArm,
    # ERRSSM_RobotArm_deterministic
    )

def get_net(in_dim, out_dim, **kwargs):
    if kwargs["arch"] == "fc_vec":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = FC_vec(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "fc_image":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        out_chan_num = kwargs["out_chan_num"]
        net = FC_image(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
            out_chan_num=out_chan_num
        )
    elif kwargs["arch"] == "conv28":
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = ConvNet28(
            in_chan=in_dim,
            out_chan=out_dim,
            activation=activation,
            out_activation=out_activation
        )
    elif kwargs["arch"] == "dconv28":
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = DeConvNet28(
            in_chan=in_dim,
            out_chan=out_dim,
            activation=activation,
            out_activation=out_activation
        )
    elif kwargs["arch"] == "conv64":
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = ConvNet64(
            in_chan=in_dim,
            out_chan=out_dim,
            activation=activation,
            out_activation=out_activation
        )
    elif kwargs["arch"] == "dconv64":
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = DeConvNet64(
            in_chan=in_dim,
            out_chan=out_dim,
            activation=activation,
            out_activation=out_activation
        )
    return net

def get_ae(**model_cfg):
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']
    arch = model_cfg["arch"]
    
    if arch == "ae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = AE(encoder, decoder)
        
    elif arch == "ggae":
        iso_reg = model_cfg.get("iso_reg", 1.0)
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = GGAE(encoder, decoder, iso_reg=iso_reg)
    
    elif arch == "timeseriesae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = TIMESERIES_AE(encoder, decoder)
    
    elif arch == "timeseriesggae":
        iso_reg = model_cfg.get("iso_reg", 1.0)
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = TIMESERIES_GGAE(encoder, decoder, iso_reg=iso_reg)
    
    return model

def get_rssm(**model_cfg):
    o_dim = model_cfg['o_dim']
    lstate_dim = model_cfg['lstate_dim']
    hidden_dim = model_cfg['hidden_dim']
    hstate_dim = model_cfg['hstate_dim']
    activation = model_cfg['activation']
    arch = model_cfg["arch"]
    pretrained = model_cfg.get("pretrained", None)
    rollout_obs_num = model_cfg.get("rollout_obs_num", 1)
    
    if arch == 'rssm':
        encoder = get_net(o_dim, lstate_dim*2, **model_cfg["encoder"])      # encoder returns [mean, std]
        decoder = get_net(lstate_dim, o_dim, **model_cfg["decoder"])
        model = RecurrentStateSpaceModel(encoder, decoder, hstate_dim=hstate_dim, hidden_dim=hidden_dim, activation=activation)
        
    # elif arch == 'isorssm':
    #     metric = model_cfg.get("metric", "identity")
    #     iso_reg = model_cfg.get("iso_reg", 1.0)
        
    #     encoder = get_net(o_dim, lstate_dim*2, **model_cfg["encoder"])
    #     decoder = get_net(lstate_dim, o_dim, **model_cfg["decoder"])
    #     model = IsoRSSM(encoder, decoder, hstate_dim=hstate_dim, hidden_dim=hidden_dim, activation=activation, iso_reg=iso_reg, metric=metric)      
        
    # elif arch == 'nrrssm':
    #     k = model_cfg.get('k', 4)
    #     approx_order = model_cfg.get('approx_order', 1)
        
    #     encoder = get_net(o_dim, lstate_dim*2, **model_cfg["encoder"])
    #     decoder = get_net(lstate_dim, o_dim, **model_cfg["decoder"])
    #     model = NRRSSM(encoder, decoder, hstate_dim=hstate_dim, hidden_dim=hidden_dim, activation=activation, k=k, approx_order=approx_order)
        
    # elif arch == 'rssm_pwh':
    #     encoder = get_net(o_dim, hidden_dim, **model_cfg["encoder"])
    #     decoder = get_net(lstate_dim, o_dim, **model_cfg["decoder"])
        
    #     model = RSSM_posterior_w_hidden(encoder, decoder, hstate_dim=hstate_dim, hidden_dim=hidden_dim, activation=activation)
        
    # elif arch == 'errssm':
    #     iso_reg = model_cfg.get('iso_reg', 1.0)
    #     bdry_reg = model_cfg.get('bdry_reg', 0.1)
    #     measure = model_cfg.get('measure', 'irvae')
    #     bdry_measure = model_cfg.get('bdry_measure', 'time_bdry')
    #     distfunc = model_cfg.get('distfunc', 'Euclidean') 
    #     bandwidth = model_cfg.get('bandwidth', 2)
        
    #     if pretrained == None:
    #         encoder = get_net(o_dim, lstate_dim*2, **model_cfg["encoder"])
    #         decoder = get_net(lstate_dim, o_dim, **model_cfg["decoder"])
    #     else:   # path to a pretrained autoencoder is given
    #         print("Pretrained encoder and decoder used...")
    #         pretrained_autoencoder = torch.load(pretrained)
    #         encoder = pretrained_autoencoder.encoder
    #         decoder = pretrained_autoencoder.decoder
        
    #     model = ERRSSM(encoder, decoder, hstate_dim=hstate_dim, hidden_dim=hidden_dim, activation=activation, iso_reg=iso_reg, bdry_reg=bdry_reg, measure=measure, bdry_measure=bdry_measure, bandwidth=bandwidth)
    
    elif arch == 'rssm_det':        
        if pretrained == None:
            encoder = get_net(o_dim, lstate_dim, **model_cfg["encoder"])    # encoder only returns mean, not std
            decoder = get_net(lstate_dim, o_dim, **model_cfg["decoder"])
        else:   # path to a pretrained autoencoder is given
            print("Pretrained encoder and decoder used...")
            pretrained_autoencoder = torch.load(pretrained)
            encoder = pretrained_autoencoder.encoder
            decoder = pretrained_autoencoder.decoder
        
        model = RSSM_deterministic(encoder, decoder, hstate_dim=hstate_dim, hidden_dim=hidden_dim, activation=activation, rollout_obs_num=rollout_obs_num)
        
    # elif arch == 'tncrssm':
    #     tnc_reg = model_cfg.get('tnc_reg', 1)
    #     neighbor_dist = model_cfg.get('neighbor_dist', 3)
        
    #     encoder = get_net(o_dim, lstate_dim*2, **model_cfg["encoder"])
    #     decoder = get_net(lstate_dim, o_dim, **model_cfg["decoder"])
        
    #     model = TNCRSSM(encoder, decoder, hstate_dim=hstate_dim, hidden_dim=hidden_dim, activation=activation, tnc_reg=tnc_reg, neighbor_dist=neighbor_dist)
    
    # elif arch == 'errssm_det':  
    #     iso_reg = model_cfg.get('iso_reg', 1.0)
    #     bdry_reg = model_cfg.get('bdry_reg', 0.0)
    #     measure = model_cfg.get('measure', 'irvae') 
    #     bdry_measure = model_cfg.get('bdry_measure', 'time_bdry') 
    #     distfunc = model_cfg.get('distfunc', 'Euclidean') 
    #     bandwidth = model_cfg.get('bandwidth', 50)     
    #     if pretrained == None:
    #         encoder = get_net(o_dim, lstate_dim, **model_cfg["encoder"])    # encoder only returns mean, not std
    #         decoder = get_net(lstate_dim, o_dim, **model_cfg["decoder"])
    #     else:   # path to a pretrained autoencoder is given
    #         print("Pretrained encoder and decoder used...")
    #         pretrained_autoencoder = torch.load(pretrained)
    #         encoder = pretrained_autoencoder.encoder
    #         decoder = pretrained_autoencoder.decoder
        
    #     model = ERRSSM_deterministic(encoder, decoder, hstate_dim=hstate_dim, hidden_dim=hidden_dim, activation=activation, iso_reg=iso_reg, bdry_reg=bdry_reg, measure=measure, bdry_measure=bdry_measure, bandwidth=bandwidth)
        
    # elif arch == 'errssm_robotarm':
    #     iso_reg = model_cfg.get('iso_reg', 1.0)
    #     measure = model_cfg.get('measure', 'irvae')
    #     bandwidth = model_cfg.get('bandwidth', 2)
        
    #     encoder = get_net(o_dim, lstate_dim*2, **model_cfg["encoder"])
    #     decoder = get_net(lstate_dim, o_dim, **model_cfg["decoder"])
        
    #     model = ERRSSM_RobotArm(encoder, decoder, hstate_dim=hstate_dim, hidden_dim=hidden_dim, activation=activation, iso_reg=iso_reg, measure=measure, bandwidth=bandwidth)
        
    # elif arch == 'errssm_robotarm_det':
    #     iso_reg = model_cfg.get('iso_reg', 1.0)
    #     measure = model_cfg.get('measure', 'irvae')
    #     bandwidth = model_cfg.get('bandwidth', 2)
        
    #     encoder = get_net(o_dim, lstate_dim, **model_cfg["encoder"])
    #     decoder = get_net(lstate_dim, o_dim, **model_cfg["decoder"])
        
    #     if pretrained is not None:
    #         pretrainedAE_state_dict = torch.load(pretrained)['model_state']
    #         encoder_state_dict = OrderedDict()
    #         decoder_state_dict = OrderedDict()
    #         for key in pretrainedAE_state_dict.keys():
    #             if 'encoder' in key:
    #                 encoder_state_dict[key[8:]] = pretrainedAE_state_dict[key]
    #             elif 'decoder' in key:
    #                 decoder_state_dict[key[8:]] = pretrainedAE_state_dict[key]
                    
    #         encoder.load_state_dict(encoder_state_dict)
    #         decoder.load_state_dict(decoder_state_dict)
            
    #     model = ERRSSM_RobotArm_deterministic(encoder, decoder, hstate_dim=hstate_dim, hidden_dim=hidden_dim, activation=activation, iso_reg=iso_reg, measure=measure, bandwidth=bandwidth)
        
    return model

def get_model(cfg, *args, version=None, **kwargs):
    # cfg can be a whole config dictionary or a value of a key 'model' in the config dictionary (cfg['model']).
    if "model" in cfg:
        model_dict = cfg["model"]
    elif "arch" in cfg:
        model_dict = cfg
    else:
        raise ValueError(f"Invalid model configuration dictionary: {cfg}")
    name = model_dict["arch"]
    model = _get_model_instance(name)
    model = model(**model_dict)
    return model

def _get_model_instance(name):
    try:
        return {
            "ae": get_ae,
            "ggae": get_ae,
            "timeseriesae": get_ae,
            "timeseriesggae": get_ae,
            
            "rssm": get_rssm,
            # "isorssm": get_rssm,
            # "nrrssm": get_rssm,
            # "rssm_pwh": get_rssm,
            # "errssm": get_rssm,
            "rssm_det": get_rssm,
            # "errssm_det": get_rssm,
            # "tncrssm": get_rssm,
            # "errssm_robotarm": get_rssm,
            # "errssm_robotarm_det": get_rssm,
        }[name]
    except:
        raise ("Model {} not available".format(name))

def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)
    
    model = get_model(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)
    
    return model, cfg
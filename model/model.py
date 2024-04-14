import logging
from collections import OrderedDict
#import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import model.networks as networks
import model.networkHelper as Helpernetwork
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.netH = self.set_device(Helpernetwork.MPRfusion())
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            self.netH.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info('Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                #optim_params = list(self.netG.parameters()) + list(self.netH.parameters())
                optim_params = list(self.netG.parameters())


            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        self.output, self.stage1_output, self.out_T, self.out_A, self.out_I = self.netH((self.data['SR'] + 1.0) / 2.0)
        condition = torch.cat([self.output / 0.5 - 1, self.out_T / 0.5 - 1], dim=1)
        l_pix = self.netG(self.data['HR'], condition)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()
        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        self.netH.eval()
        with torch.no_grad():
            self.output, self.stage1_output, self.out_T, self.out_A, self.out_I = self.netH((self.data['SR'] + 1.0) / 2.0)
            condition = torch.cat([self.output/0.5 - 1, self.out_T/0.5 - 1], dim=1)
            self.SR = self.netG.super_resolution(condition)

        self.netG.train()
        self.netH.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['Out'] = torch.clamp((self.SR + 1.0) / 2.0, min=0.0, max=1.0).detach().float().cpu()
            out_dict['LR'] = torch.clamp((self.data['SR'] + 1.0) / 2.0, min=0.0, max=1.0).detach().float().cpu()
            out_dict['HR'] = torch.clamp((self.data['HR'] + 1.0) / 2.0, min=0.0, max=1.0).detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        sH, nH = self.get_network_description(self.netH)
        if isinstance(self.netH, nn.DataParallel):
            net_struc_strH = '{} - {}'.format(self.netH.__class__.__name__, self.netH.module.__class__.__name__)
        else:
            net_struc_strH = '{}'.format(self.netH.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info('Network H structure: {}, with parameters: {:,d}'.format(net_struc_strH, nH))
        #logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info('Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_pathG = self.opt['path']['resume_state']
        if load_pathG is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_pathG))
            gen_path = '{}_gen.pth'.format(load_pathG)
            #opt_path = '{}_opt.pth'.format(load_pathG)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            net = torch.load(gen_path)
            net.pop("denoise_fn.downs.0.weight")
            network.load_state_dict(net, strict=False)
            #network.load_state_dict(torch.load(gen_path), strict=(not self.opt['model']['finetune_norm']))
            
            
            # network.load_state_dict(torch.load(gen_path), strict=False)
            #if self.opt['phase'] == 'train':
            #    # optimizer
            #    opt = torch.load(opt_path)
            #    self.optG.load_state_dict(opt['optimizer'])
            #    self.begin_step = opt['iter']
            #    self.begin_epoch = opt['epoch']

        load_pathH = self.opt['path']['resume_stateH']
        if load_pathH is not None:
            logger.info('Loading pretrained model for H [{:s}] ...'.format(load_pathH))
            network = self.netH
            load_net = torch.load(load_pathH, map_location=lambda storage, loc: storage)
            load_net = load_net['params']
            # load_net = load_net['model']
            # remove unnecessary 'module.'
            for k, v in deepcopy(load_net).items():
                if k.startswith('module.'):
                    load_net[k[7:]] = v
                    load_net.pop(k)
            network.load_state_dict(load_net, strict=(not self.opt['model']['finetune_norm']))




# class DDPM(BaseModel):
#     def __init__(self, opt):
#         super(DDPM, self).__init__(opt)
#         # define network and load pretrained models
#         self.netG = self.set_device(networks.define_G(opt))
#         self.schedule_phase = None
#
#         # set loss and load resume state
#         self.set_loss()
#         self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
#         if self.opt['phase'] == 'train':
#             self.netG.train()
#             # find the parameters to optimize
#             if opt['model']['finetune_norm']:
#                 optim_params = []
#                 for k, v in self.netG.named_parameters():
#                     v.requires_grad = False
#                     if k.find('transformer') >= 0:
#                         v.requires_grad = True
#                         v.data.zero_()
#                         optim_params.append(v)
#                         logger.info('Params [{:s}] initialized to 0 and will optimize.'.format(k))
#             else:
#                 optim_params = list(self.netG.parameters())
#
#             self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"])
#             self.log_dict = OrderedDict()
#         self.load_network()
#         self.print_network()
#
#     def feed_data(self, data):
#         self.data = self.set_device(data)
#
#     def optimize_parameters(self):
#         self.optG.zero_grad()
#         l_pix = self.netG(self.data)
#         # need to average in multi-gpu
#         b, c, h, w = self.data['HR'].shape
#         l_pix = l_pix.sum()/int(b*c*h*w)
#         l_pix.backward()
#         self.optG.step()
#         # set log
#         self.log_dict['l_pix'] = l_pix.item()
#
#     def test(self, continous=False):
#         self.netG.eval()
#         with torch.no_grad():
#             if isinstance(self.netG, nn.DataParallel):
#                 self.SR = self.netG.module.super_resolution(self.data, continous)
#             else:
#                 self.SR = self.netG.super_resolution(self.data, continous)
#         self.netG.train()
#
#     def sample(self, batch_size=1, continous=False):
#         self.netG.eval()
#         with torch.no_grad():
#             if isinstance(self.netG, nn.DataParallel):
#                 self.SR = self.netG.module.sample(batch_size, continous)
#             else:
#                 self.SR = self.netG.sample(batch_size, continous)
#         self.netG.train()
#
#     def set_loss(self):
#         if isinstance(self.netG, nn.DataParallel):
#             self.netG.module.set_loss(self.device)
#         else:
#             self.netG.set_loss(self.device)
#
#     def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
#         if self.schedule_phase is None or self.schedule_phase != schedule_phase:
#             self.schedule_phase = schedule_phase
#             if isinstance(self.netG, nn.DataParallel):
#                 self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
#             else:
#                 self.netG.set_new_noise_schedule(schedule_opt, self.device)
#
#     def get_current_log(self):
#         return self.log_dict
#
#     def get_current_visuals(self, need_LR=True, sample=False):
#         out_dict = OrderedDict()
#         if sample:
#             out_dict['SAM'] = self.SR.detach().float().cpu()
#         else:
#             out_dict['Out'] = self.SR.detach().float().cpu()
#             out_dict['LR'] = self.data['SR'].detach().float().cpu()
#             out_dict['HR'] = self.data['HR'].detach().float().cpu()
#         return out_dict
#
#     def print_network(self):
#         s, n = self.get_network_description(self.netG)
#         if isinstance(self.netG, nn.DataParallel):
#             net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
#         else:
#             net_struc_str = '{}'.format(self.netG.__class__.__name__)
#
#         logger.info(
#             'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
#         logger.info(s)
#
#     def save_network(self, epoch, iter_step):
#         gen_path = os.path.join(
#             self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
#         opt_path = os.path.join(
#             self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
#         # gen
#         network = self.netG
#         if isinstance(self.netG, nn.DataParallel):
#             network = network.module
#         state_dict = network.state_dict()
#         for key, param in state_dict.items():
#             state_dict[key] = param.cpu()
#         torch.save(state_dict, gen_path)
#         # opt
#         opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
#         opt_state['optimizer'] = self.optG.state_dict()
#         torch.save(opt_state, opt_path)
#
#         logger.info('Saved model in [{:s}] ...'.format(gen_path))
#
#     def load_network(self):
#         load_path = self.opt['path']['resume_state']
#         if load_path is not None:
#             logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path))
#             gen_path = '{}_gen.pth'.format(load_path)
#             opt_path = '{}_opt.pth'.format(load_path)
#             # gen
#             network = self.netG
#             if isinstance(self.netG, nn.DataParallel):
#                 network = network.module
#             network.load_state_dict(torch.load(gen_path), strict=(not self.opt['model']['finetune_norm']))
#             # network.load_state_dict(torch.load(
#             #     gen_path), strict=False)
#             if self.opt['phase'] == 'train':
#                 # optimizer
#                 opt = torch.load(opt_path)
#                 self.optG.load_state_dict(opt['optimizer'])
#                 self.begin_step = opt['iter']
#                 self.begin_epoch = opt['epoch']

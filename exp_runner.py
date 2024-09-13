import os
import time
import logging
import argparse
import numpy as np
import cv2
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
import random
import imageio
from pyhocon import ConfigFactory
from datasets.shapenet import ShapeNetDataset
from datasets.dtu import DTUDataset
from datasets.replica import ReplicaDataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.ray2weight import *
from extract_mesh import get_mesh_udf_fast




class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset_type = self.conf['dataset.type']
        if self.dataset_type == 'shapenet':
            self.dataset = ShapeNetDataset(self.conf['dataset'])
        if self.dataset_type in ['dtu', 'deepfashion', 'real_captured']:
            self.dataset = DTUDataset(self.conf['dataset'])
        elif self.dataset_type == 'replica':
            self.dataset = ReplicaDataset(self.conf['dataset'])
        else:
            raise NotImplementedError('No matched dataset type.')

        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.window_size = self.conf.get_int('model.window_size', default=11)
        mlp_type = self.conf.get_string('model.mlp_type', default='udfxsdf')
        self.neus_init_end = self.conf.get_int('train.neus_init_end', default=0)
        self.mask_init_end = self.conf.get_int('train.mask_init_end', default=0)


        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.smooth_weight = self.conf.get_float('train.smooth_weight', default=0.0)
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        other_params = []   # sdf_param, color_param, other_param(nerf_outside, alphamlp, s)
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)  # sdf net
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)  # train for parameter s in paper
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)  # color net, same as nerf
        self.alphatrans_network = Ray2Alpha(self.window_size, init_bias=('udfxsdf' in mlp_type))

        params_to_train += list(self.nerf_outside.parameters())
        other_params += list(self.nerf_outside.parameters())
        if 'train_mlp' in self.mode:
            params_to_train += list(self.alphatrans_network.parameters())
            other_params += list(self.alphatrans_network.parameters())
        if 'train_udf' in self.mode:
            params_to_train += list(self.sdf_network.parameters())
            sdf_params = list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        other_params += list(self.deviation_network.parameters())
        if 'color' in self.mode:
            params_to_train += list(self.color_network.parameters())
            color_params = list(self.color_network.parameters())

        if 'train_mlp' in self.mode:
            self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate, weight_decay=1e-4)
        elif 'train_udf_color' in self.mode:
            self.optimizer = torch.optim.Adam([{'params': sdf_params, 'lr': self.learning_rate},
                                               {'params': other_params, 'lr': self.learning_rate},
                                               {'params': color_params, 'lr': self.learning_rate}])
        elif 'train_udf' in self.mode:
            self.optimizer = torch.optim.Adam([{'params': sdf_params, 'lr': self.learning_rate * 1.0},
                                               {'params': other_params, 'lr': self.learning_rate},])

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.alphatrans_network,
                                     **self.conf['model.neus_renderer'])
        self.renderer.mode = self.mode
        self.renderer.window_size = self.window_size
        self.renderer.mlp_type = mlp_type
        self.renderer.dataset_type = self.dataset_type
        self.renderer.neus_init_end = self.neus_init_end

        # Load volume rendering prior checkpoint
        latest_model_name = None
        if 'train_mlp' in self.mode:
            pass
        elif 'train_udf' in self.mode and 'validate_mesh' not in self.mode:
            if mlp_type == 'udfxsdf':      # dtu, replica
                latest_model_name = 'log/mlp/shapenet_df3d_udfxsdf/checkpoints/ckpt_060000.pth'
            elif mlp_type == 'udfxdist':        # deepfashion, real_captured
                if self.dataset_type == 'real_captured':
                    latest_model_name = 'log/mlp/shapenet_df3d_udfxdist/checkpoints/ckpt_020000.pth'
                else:
                    latest_model_name = 'log/mlp/shapenet_df3d_udfxdist/checkpoints/ckpt_100000.pth'
            else:
                raise NotImplementedError('no match pretrained mlp type.')
            print(latest_model_name)

            ch = torch.load(latest_model_name, map_location=self.device)
            self.alphatrans_network.load_state_dict(ch['alphamlp'])

        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth':
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            print('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name, is_continue=is_continue)

        if 'train' in self.mode[:5] and 'validate_mesh' not in self.mode and 'validate_video' not in self.mode:
            self.file_backup()

        if 'train_mlp' in self.mode:
            sdf_volume0 = np.load('data/prior_datasets/shapenet_02958343/model_volume_1024.npy')
            sdf_volume0 = torch.tensor(sdf_volume0)
            sdf_volume1 = np.load('data/prior_datasets/df3d_1/model_volume_1024.npy')
            sdf_volume1 = torch.tensor(sdf_volume1)

            self.renderer.sdf_network.sdf_volume = [sdf_volume0[None, None, ...], sdf_volume1[None, None, ...]]
            self.renderer.sdf_network.bound_scale = 2.0

        self.renderer.sdf_network.isudf = True
        if mlp_type == 'udfxsdf':
            self.renderer.ray2alpha.sharpness = 20.0
        elif mlp_type == 'udfxdist':
            self.renderer.ray2alpha.sharpness = 1.0


    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.renderer.base_exp_dir = self.base_exp_dir
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        if self.mode != 'train_mlp':
            self.validate_image(resolution_level=8)
        else:
            self.validate_depth_image()
        if self.renderer.sdf_network.isudf and self.iter_step > 100:
            try:
                self.extract_udf_mesh()
            except:
                print('try udf extraction failed. use sdf extraction instead.')
                self.validate_mesh(threshold=0.005)
        else:
            self.validate_mesh()

        for iter_i in tqdm(range(res_step)):

            img_idx = image_perm[self.iter_step % len(image_perm)]
            data = self.dataset.gen_random_rays_at(img_idx, self.batch_size)

            rays_o, rays_d, true_rgb, true_depth, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, 10:11]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
            self.renderer.true_depth = true_depth
            scene_id = self.dataset.get_scene_id(img_idx)   # always 0 for mode!='train_mlp'
            self.renderer.sdf_network.scene_id = scene_id
            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            gradient_error = render_out['gradient_error']
            weight_sum = render_out['weight_sum']
            depth = render_out['depth']
            depth_mask = render_out['depth_mask']

            # Loss
            if 'color' in self.mode:
                if self.dataset_type in ['dtu', 'deepfashion', 'real_captured', 'replica', 'Blender']:
                    color_error = (color_fine - true_rgb) * mask
                    color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())
            else:
                color_fine_loss = 0

            if 'train_mlp' in self.mode:
                eikonal_loss = 0
            elif 'train_udf' in self.mode:
                eikonal_loss = gradient_error

            if self.dataset_type in ['dtu', 'deepfashion', 'real_captured', 'replica'] and 'wodepth' in self.mode:
                if self.iter_step < self.mask_init_end and self.dataset_type == 'deepfashion':
                    mask = (true_rgb>1e-5).any(1, keepdim=True).float()     # add black bg mask for better convergence at early stage
                    mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
                elif self.dataset_type == 'replica':        # replica mask=1 for all pixels
                    mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
                else:
                    mask_loss = 0.
                depth_loss = 0
            elif 'train_mlp' not in self.mode and 'wodepth' in self.mode:
                depth_loss = 0.
                mask_loss = 0.
            elif 'train_mlp' in self.mode:
                depth_mask = true_depth != 0
                if self.renderer.mlp_type == 'udfxsdf':
                    soft_thres = 0.02
                    if self.iter_step > 20000:
                        soft_thres = 0.0001
                    relax_no_inter_mask = render_out['ray_info'][1].min(1)[0] > soft_thres
                    relax_inter_mask = render_out['ray_info'][1].min(1)[0] < -soft_thres
                    relax_mask = torch.bitwise_or(relax_no_inter_mask, relax_inter_mask)
                    depth_loss = (torch.abs(depth - true_depth))[relax_mask].mean()
                    mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3)[relax_mask], depth_mask.float()[relax_mask])
                elif self.renderer.mlp_type == 'udfxdist':
                    relax_no_inter_mask = render_out['ray_info'][1].abs().min(1)[0] > 0.01  # for window11
                    relax_mask = torch.bitwise_or(relax_no_inter_mask, depth_mask.squeeze())
                    depth_loss = (torch.abs(depth - true_depth))[depth_mask].mean()
                    mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), depth_mask.float())
                    if self.iter_step > 10000:  # small mask_weight at start
                        self.mask_weight = 1.0

            else:   # mode=='train_udf'
                depth_mask = true_depth != 0
                depth_loss = (torch.abs(depth - true_depth))[depth_mask].mean()
                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), depth_mask.float())


            if 'wodepth' in self.mode:
                depth_weight = 0.
            else:
                depth_weight = 1.0

            if 'train_udf' in self.mode:
                # g1 = render_out['gradients'].reshape(-1, 3)
                g1 = render_out['surface_grad'].reshape(-1, 3)
                g1_norm = g1 / (g1.norm(2, dim=1, keepdim=True) + 1e-5)
                g2 = render_out['neighbor_grad'].reshape(-1, 3)
                g2_norm = g2 / (g2.norm(2, dim=1, keepdim=True) + 1e-5)
                assert g1.shape == g2.shape, 'surface grad and neighbor grad not match!'
                if g1.shape[0] != 0:
                    smooth_loss = torch.norm(g1_norm - g2_norm, dim=-1).mean()
                else:
                    smooth_loss = 0.
            else:
                smooth_loss = 0.


            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight + \
                   depth_loss * depth_weight + \
                   smooth_loss * self.smooth_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # real_captured switch harder mlp
            if self.iter_step == 50000 and self.dataset_type == 'real_captured':
                print('\nswitch to unbiased twomlp MLP...\n')
                latest_model_name = 'log/mlp/shapenet_df3d_udfxdist/checkpoints/ckpt_100000.pth'
                ch = torch.load(latest_model_name, map_location=self.device)
                self.alphatrans_network.load_state_dict(ch['alphamlp'])

            self.iter_step += 1
            self.renderer.iter_step = self.iter_step

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Loss/depth_loss', depth_loss, self.iter_step)
            self.writer.add_scalar('Loss/smooth_loss', smooth_loss, self.iter_step)
            self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
            self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.iter_step)
            self.writer.add_scalar('Train/sharpness', self.renderer.ray2alpha.sharpness, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss={:.3f} color={:.3f} depth={:.3f} mask={:.3f} eik={:.3f} smooth={:.3f} mask_sum={:d}'
                    .format(self.iter_step, loss, color_fine_loss, depth_loss, mask_loss, eikonal_loss, smooth_loss, depth_mask.sum()))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0 and 'train_mlp' in self.mode:
                self.validate_depth_image()

            if self.iter_step % (self.val_freq*2) == 0 and 'train_udf_color' in self.mode:
                self.validate_image(resolution_level=self.validate_resolution_level)

            if self.iter_step % self.val_mesh_freq == 0 and 'train_udf' in self.mode:
                if self.renderer.sdf_network.isudf:
                    try:
                        self.extract_udf_mesh()
                    except:
                        print('udf extraction failed. try sdf extraction.')
                        self.validate_mesh(threshold=0.005)

                else:
                    self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()


    def get_image_perm(self):
        return torch.randperm(self.dataset.__len__())

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        if 'train_mlp' in self.mode:
            for g in self.optimizer.param_groups:
                g['lr'] = self.learning_rate * learning_factor
        elif 'train_udf' in self.mode:
            for g in self.optimizer.param_groups[:1]:   # sdf
                # real_captured: fix sdf at first 1000 epoch
                if self.dataset_type == 'real_captured' and self.iter_step < 1000:
                    g['lr'] = self.learning_rate * learning_factor * 0.
                else:
                    g['lr'] = self.learning_rate * learning_factor * 1.0
            for g in self.optimizer.param_groups[1:2]:  # other
                g['lr'] = self.learning_rate * learning_factor
            if 'train_udf_color' in self.mode:          # inside color
                for g in self.optimizer.param_groups[2:]:
                    g['lr'] = self.learning_rate * learning_factor * 1.0

    def file_backup(self):
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        os.system("""cp -r {0} "{1}/" """.format('confs', os.path.join(self.base_exp_dir, 'recording')))
        os.system("""cp -r {0} "{1}/" """.format('datasets', os.path.join(self.base_exp_dir, 'recording')))
        os.system("""cp -r {0} "{1}/" """.format('models', os.path.join(self.base_exp_dir, 'recording')))
        os.system("""cp {0} "{1}/" """.format('exp_runner.py', os.path.join(self.base_exp_dir, 'recording')))
        os.system("""cp {0} "{1}/" """.format('extract_mesh.py', os.path.join(self.base_exp_dir, 'recording')))

    def load_checkpoint(self, checkpoint_name, is_continue=False):
        if is_continue:
            checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                    map_location=self.device)
            self.nerf_outside.load_state_dict(checkpoint['nerf'])
            self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
            self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
            self.color_network.load_state_dict(checkpoint['color_network_fine'])
            if self.mode == 'train_mlp':
                self.alphatrans_network.load_state_dict(checkpoint['alphamlp'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.iter_step = checkpoint['iter_step']
            # self.iter_step = 0
            logging.info('End')
            return

        if 'train_mlp' in self.mode:
            # checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
            checkpoint = torch.load(checkpoint_name, map_location=self.device)
            # self.nerf_outside.load_state_dict(checkpoint['nerf'])
            self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
            self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
            self.alphatrans_network.load_state_dict(checkpoint['alphamlp'])
            # self.color_network.load_state_dict(checkpoint['color_network_fine'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.iter_step = checkpoint['iter_step']
            logging.info('End')
            return
        elif self.mode == 'validate_mesh':
            checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
            self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
            self.iter_step = checkpoint['iter_step']
            logging.info('End')
            return


    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'alphamlp': self.alphatrans_network.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_depth_image(self, idx=-1):
        self.renderer.is_validate = True
        if idx < 0:
            idx = np.random.randint(self.dataset.__len__())
        # idx = 27
        self.renderer.sdf_network.scene_id = 0
        print('Validate Depth: iter: {}, camera: {}'.format(self.iter_step, idx))

        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=1)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(1536)
        rays_d = rays_d.reshape(-1, 3).split(1536)

        out_depth_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            render_out = self.renderer.render(rays_o_batch, rays_d_batch, near, far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('depth'):
                out_depth_fine.append(render_out['depth'].cpu().detach().numpy())
            del render_out

        depth_img = np.concatenate(out_depth_fine, axis=0).reshape([H, W])
        depth_img = np.array(depth_img / depth_img.max() * 65535, dtype=np.uint16)
        write_dir = os.path.join(self.base_exp_dir, 'validations_depth')
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        cv2.imwrite(os.path.join(write_dir, '{:0>8d}_{}.png'.format(self.iter_step, idx)), depth_img)
        # cv2.imwrite(os.path.join(write_dir, '{:d}.png'.format(self.iter_step//self.val_freq)), depth_img)

        self.renderer.is_validate = False

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(len(self.dataset))

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        self.renderer.is_validate = True
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out
        self.renderer.is_validate = False
        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv2.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[:, :, [0,1,2], i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv2.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def validate_video(self):
        frames = []
        for idx in tqdm(range(len(self.dataset)-1)):
            n_frames = 5
            for i in range(n_frames):
                print(i)
                rays_o, rays_d = self.dataset.gen_rays_between(idx, idx+1,
                                                               ratio=np.sin(((i/n_frames)-0.5)*np.pi)*0.5+0.5,
                                                               resolution_level=1)
                # rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=1)
                H, W, _ = rays_o.shape
                rays_o = rays_o.reshape(-1, 3).split(1400)
                rays_d = rays_d.reshape(-1, 3).split(1400)

                out_rgb_fine = []

                self.renderer.is_validate = True
                for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                    near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                    background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
                    render_out = self.renderer.render(rays_o_batch,
                                                      rays_d_batch,
                                                      near,
                                                      far,
                                                      cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                      background_rgb=background_rgb)

                    out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                    del render_out
                img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)[:,:,[2,1,0]]
                im = (img_fine).astype(np.uint8)
                # cv2.imwrite(f'debug/{idx*n_frames+i}.jpg', im)
                frames.append(im)
            # break

        os.makedirs(os.path.join(self.base_exp_dir, 'video'), exist_ok=True)
        vid_path = os.path.join(self.base_exp_dir, 'video', 'video.mp4')
        imageio.mimwrite(vid_path, frames, fps=10, macro_block_size=8)



    def validate_mesh(self, world_space=False, resolution=128, threshold=0.0):      # TODO: threshold
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def extract_udf_mesh(self, world_space=False, resolution=256, dist_threshold_ratio=1.0, vertex_color=False):
        if not self.renderer.sdf_network.isudf:
            func = self.renderer.sdf_network

            def func_grad(xyz):
                gradients = self.renderer.sdf_network.gradient(xyz)
                gradients_mag = torch.linalg.norm(gradients, ord=2, dim=-1, keepdim=True)
                gradients_norm = gradients / (gradients_mag + 1e-5)  # normalize to unit vector
                return gradients_norm

        else:
            func = lambda pts: torch.abs(self.sdf_network.udf(pts))
            func_grad = lambda pts: self.sdf_network.gradient(pts)

        try:
            pred_v, pred_f, pred_mesh, samples, indices = get_mesh_udf_fast(func, func_grad, samples=None,
                                                                            indices=None, N_MC=resolution,
                                                                            gradient=True, eps=0.005,
                                                                            border_gradients=True,
                                                                            smooth_borders=True,
                                                                            dist_threshold_ratio=dist_threshold_ratio)
        except:
            print('border_gradients=False, smooth_borders=False')
            pred_v, pred_f, pred_mesh, samples, indices = get_mesh_udf_fast(func, func_grad, samples=None,
                                                                            indices=None, N_MC=resolution,
                                                                            gradient=True, eps=0.005,
                                                                            border_gradients=False,
                                                                            smooth_borders=False,
                                                                            dist_threshold_ratio=dist_threshold_ratio)

        vertices, triangles = pred_mesh.vertices, pred_mesh.faces
        if vertex_color:
            colors = self.renderer.obtain_vertex_color(vertices)
        else:
            colors = None
        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
        vertices /= self.dataset.scene_scale
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=colors)

        os.makedirs(os.path.join(self.base_exp_dir, 'udf_meshes'), exist_ok=True)
        mesh.export(
            os.path.join(self.base_exp_dir, 'udf_meshes', 'udf_res{}_step{}.ply'.format(resolution, self.iter_step)))


if __name__ == '__main__':
    print('Hello Wooden')

    seed = 1001
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True) # debug

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--resolution', type=int, default=512)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if 'validate_mesh' in args.mode:
        if runner.sdf_network.isudf:
            runner.extract_udf_mesh(world_space=True, resolution=args.resolution, vertex_color=False)
            # runner.validate_mesh(world_space=True, resolution=512, threshold=0.005)
        else:
            runner.validate_mesh(world_space=True, resolution=args.resolution, threshold=args.mcube_threshold)
    elif 'validate_video' in args.mode:
        runner.validate_video()
    elif 'train' in args.mode:
        runner.train()
        runner.extract_udf_mesh(world_space=True, resolution=512, vertex_color=False)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)

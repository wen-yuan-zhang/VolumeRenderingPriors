import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
import point_cloud_utils as pcu
import math
import os
import trimesh
from tqdm import tqdm


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 alphatrans_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.ray2alpha = alphatrans_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

        self.iter_step = 0
        self.window_size = 5
        self.is_validate = False

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples


    def up_sample_neudf(self, rays_o, rays_d, z_vals, sdf, n_importance, i):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        inv_s = 64 * 2 ** i

        sigma = 1 / (1 + torch.exp(-mid_sdf * inv_s))
        rho = sigma * (1 - sigma) * inv_s

        alpha = 1 - torch.exp(-rho * (next_z_vals - prev_z_vals))
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        for _ in range(3):
            weights_prev = torch.cat([weights[:, :1], weights[:, :-1]], dim=-1)
            weights_next = torch.cat([weights[:, 1:], weights[:, -1:]], dim=-1)
            weights = torch.max(weights, weights_prev)
            weights = torch.max(weights, weights_next)

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            if self.sdf_network.isudf and self.udf_use_sdf_upsamp:
                new_sdf = self.sdf_network.udf_sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            else:
                new_sdf = self.sdf_network.udf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):  # Render part in bounding sphere (core), using color net
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_network.return_sdf = True
        if 'train_mlp' in self.mode:
            if sdf_network.isudf:
                sdf, udf = sdf_network(pts, return_sdf=sdf_network.return_sdf)
            else:
                sdf = sdf_network(pts)
                udf = sdf
            gradients = sdf_network.gradient(pts).squeeze()
            feature_vector = torch.zeros(batch_size*n_samples, 256).cuda()
        else:
            if sdf_network.isudf:
                sdf, udf_nn_output = sdf_network(pts, return_sdf=True)
                udf = udf_nn_output[:, :1]
                feature_vector = udf_nn_output[:, 1:]
            else:
                sdf_nn_output = sdf_network(pts)
                sdf = sdf_nn_output[:, :1]
                udf = sdf
                feature_vector = sdf_nn_output[:, 1:]
            gradients = sdf_network.gradient(pts).squeeze()

        # flip udf gradient to sdf gradient, because udf surface gradient is unstable, not safe for color net
        if sdf_network.isudf:
            sign = sdf.reshape(-1, 1) * udf.reshape(-1, 1)
            sign = torch.where(sign > 0, torch.tensor([1.]).cuda(), torch.tensor([-1.]).cuda())
            gradients = sign * gradients

        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        sdf_norm = sdf.reshape(batch_size, n_samples)
        # for closed surface, borrow neus for initialization
        if 'train_udf' in self.mode and self.iter_step < self.neus_init_end:
            # self.sdf_network.udf_activ = torch.abs
            inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
            inv_s = inv_s.expand(batch_size * n_samples, 1)
            true_cos = (dirs * gradients).sum(-1, keepdim=True)
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio)
            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
            udf_norm = sdf.reshape(batch_size, n_samples)
        else:
            inv_s = torch.ones_like(sdf.reshape(-1, 1))
            sdf_input = self.sdf_windowseq(sdf_norm, self.window_size, window_norm=False)
            sdf_input = sdf_input.reshape(batch_size*n_samples, -1)
            if self.sdf_network.return_sdf:
                udf_norm = udf.reshape(batch_size, n_samples)
                udf_input = self.sdf_windowseq(udf_norm, self.window_size, window_norm=False)
                udf_input = udf_input.reshape(batch_size * n_samples, -1)
                # for closed surface, concat sdf(before adding 'abs') with udf for better intersection prediction
                if self.mlp_type =='udfxsdf':
                    sdf_input = torch.stack([sdf_input, udf_input], -1)
                elif self.mlp_type == 'udfxdist':
                    dist_input = self.sdf_windowseq(dists, self.window_size, window_norm=False)
                    dist_input = dist_input.reshape(batch_size * n_samples, -1)
                    sdf_input = torch.stack([udf_input, dist_input], -1)
            if 'train_udf' not in self.mode:
                sdf_input = sdf_input.detach()

            alpha = self.ray2alpha(sdf_input)
            alpha = alpha.reshape(batch_size, n_samples)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        depth = (weights * mid_z_vals).sum(-1, keepdim=True)
        depth_mask1 = (sdf.reshape(batch_size, n_samples) > 0).any(1)
        depth_mask2 = sdf.reshape(batch_size, n_samples)[:,0] > 0
        if self.sdf_network.isudf:
            depth_mask3 = (udf.reshape(batch_size, n_samples) < 0.01).any(1)
        else:
            depth_mask3 = (sdf.reshape(batch_size, n_samples) < 0).any(1)
        depth_mask = torch.bitwise_and(depth_mask1, depth_mask2)
        depth_mask = torch.bitwise_and(depth_mask, depth_mask3)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] + \
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

            weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
            weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:  # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        if self.dataset_type in ['deepfashion', 'shapenet', 'real_captured']:
            relax_inside_sphere = (pts_norm < 1.2).float().detach()
            near_surface_mask = (udf_norm.reshape(batch_size, n_samples) < 0.01).float().detach()
            error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2, dim=-1) - 1.0) ** 2
            gradient_error1 = (near_surface_mask * error).sum() / (near_surface_mask.sum() + 1e-5)
            mask2 = relax_inside_sphere * (1-near_surface_mask)
            gradient_error2 = (mask2 * error).sum() / (mask2.sum() + 1e-5)
            gradient_error = gradient_error1 * 0.1 + gradient_error2
        elif self.dataset_type in ['dtu', 'replica']:
            relax_inside_sphere = (pts_norm < 1.2).float().detach()
            gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2, dim=-1) - 1.0) ** 2
            gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        else:
            raise NotImplementedError('No matched dataset type for eikonal loss.')

        # # unisurf smooth loss
        surface_mask = (depth > 0.1).squeeze()
        surface_pts = rays_o[surface_mask] + rays_d[surface_mask] * depth[surface_mask]
        surface_grad = sdf_network.gradient(surface_pts).squeeze()
        # neighbor_pts = pts + (torch.rand_like(pts) - 0.5) * 0.01
        neighbor_pts = surface_pts + (torch.rand_like(surface_pts) - 0.5) * 0.01
        neighbor_grad = sdf_network.gradient(neighbor_pts).squeeze()

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            'surface_grad': surface_grad,
            'neighbor_grad': neighbor_grad,
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'depth': depth,
            'depth_mask': depth_mask,
            'ray_info': [mid_z_vals, sdf_norm.reshape(batch_size, n_samples), alpha, pts],
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_length = 3.0 if self.dataset_type=='replica' else 2.0
        sample_dist = sample_length / self.n_samples
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * sample_length / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        norm_scale = rays_d.norm(2, 1, keepdim=True)
        rays_d /= norm_scale
        self.norm_scale = norm_scale

        # Up sample
        if self.mlp_type == 'udfxsdf':
            self.udf_use_sdf_upsamp = True
        elif self.mlp_type == 'udfxdist':
            self.udf_use_sdf_upsamp = False
        else:
            raise NotImplementedError('not matched mlp type.')
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                if self.udf_use_sdf_upsamp:
                    sdf = self.sdf_network.udf_sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                else:
                    sdf = self.sdf_network.udf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    if not self.sdf_network.isudf:
                        new_z_vals = self.up_sample(rays_o, rays_d, z_vals, sdf,
                                                    self.n_importance // self.up_sample_steps, 64 * 2**i)
                    elif self.sdf_network.isudf:
                        if self.udf_use_sdf_upsamp:
                            new_z_vals = self.up_sample(rays_o, rays_d, z_vals, sdf,
                                                        self.n_importance // self.up_sample_steps, 64 * 2 ** i)
                        else:
                            new_z_vals = self.up_sample_neudf(rays_o, rays_d, z_vals, sdf,
                                                             self.n_importance // self.up_sample_steps, i)
                    z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']

        return {
            'color_fine': color_fine,
            'depth': ret_fine['depth'] / norm_scale,
            'depth_mask': ret_fine['depth_mask'],
            's_val': ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True),
            # 'cdf_fine': ret_fine['cdf'],　　
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'surface_grad': ret_fine['surface_grad'],
            'neighbor_grad': ret_fine['neighbor_grad'],
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'ray_info': ret_fine['ray_info'],
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: self.sdf_network.udf_sdf(pts))

    def sdf_windowseq(self, sdf, wsize=11, window_norm=False, window_offset=0):
        # sdf: [batch_size, n_samples]
        # return: [batch_size, n_samples, wsize]
        sdf_padded = sdf
        for i in range(wsize // 2 + window_offset):
            sdf_padded = torch.cat([sdf[:, :1], sdf_padded], 1)
        for i in range(wsize // 2 - window_offset):
            sdf_padded = torch.cat([sdf_padded, sdf[:, -1:]], 1)
        sdf_seq = []
        for i in range(wsize):
            if i != wsize-1:
                sdf_seq.append(sdf_padded[:, i:-wsize+i+1])
            else:
                sdf_seq.append(sdf_padded[:, i:])
        sdf_seq = torch.stack(sdf_seq, -1)
        if window_norm:
            window_max = torch.max(sdf_seq.abs(), -1, keepdim=True)[0] + 1e-6
            sdf_seq = sdf_seq / window_max
        return sdf_seq

    def obtain_vertex_color(self, pts):
        # pts: [N, 3]
        vertices_batch = torch.split(torch.tensor(pts, dtype=torch.float32).cuda(), 50000)
        vertex_colors = []
        for vertices in tqdm(vertices_batch):
            feature_vector = self.sdf_network.sdf_hidden_appearance(vertices)
            gradients = self.sdf_network.gradient(vertices).squeeze()
            dirs = -gradients
            vertex_color = self.color_network(vertices, gradients, dirs,
                                              feature_vector).detach().cpu().numpy()[..., ::-1]  # BGR to RGB
            vertex_colors.append(vertex_color)
        vertex_colors = np.concatenate(vertex_colors)
        print(f'validate point count: {vertex_colors.shape[0]}')
        return vertex_colors
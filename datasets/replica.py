import cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from torchvision import transforms as T
from .ray_utils import *

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
scale_dict = {
    # Replica
    'office0': 0.4,
    'office1': 0.41,
    'office2': 0.24,
    'office3': 0.21,
    'office4': 0.30,
    'room0': 0.25,
    'room1': 0.30,
    'room2': 0.29,
}

class ReplicaDataset(Dataset):
    def __init__(self, conf, split='train', N_vis=-1):
        self.device = torch.device('cuda')
        self.N_vis = N_vis
        self.root_dir = conf.get_string('data_dir')
        scene = conf.get_string('scene')

        self.split = split
        self.is_stack = False
        self.downsample = 1.0
        self.transform = T.ToTensor()
        self.scene_scale = 1.0
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_depth = []
        self.directions = []
        self.read_meta(scene)

        self.white_bg = True

        # for Neus exp_runner
        self.pose_all = self.poses
        self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])
        self.scale_mats_np = [np.eye(4)]

    def read_meta(self, scene):
        root_dir = os.path.join(self.root_dir, scene)
        with open(os.path.join(root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = int(self.meta['w'] / self.downsample), int(self.meta['h'] / self.downsample)
        self.img_wh = [w, h]
        self.focal_x = 0.5 * w / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal_y = 0.5 * h / np.tan(0.5 * self.meta['camera_angle_y'])  # original focal length
        self.cx, self.cy = self.meta['cx'], self.meta['cy']

        # ray directions for all pixels, same for all images (same H, W, focal)
        direction = get_ray_directions(h, w, [self.focal_x, self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        self.directions.append(direction)
        # self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x, 0, self.cx], [0, self.focal_y, self.cy], [0, 0, 1]]).float()

        idxs = list(range(0, len(self.meta['frames'])))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):  # img_list:#
            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix'])
            pose = pose @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses.append(c2w)
            image_path = os.path.join(root_dir, f"{frame['file_path']}")
            image_id = int(os.path.basename(image_path)[:-4])
            img = Image.open(image_path)
            img = self.transform(img)  # (4, h, w)
            img = img.permute(1, 2, 0)  # (h*w, 4) RGBA
            if img.shape[-1] == 4:
                img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])  # blend A to RGB
            self.all_rgbs.append(img)


        self.poses = torch.stack(self.poses)  #(N, 4, 4)
        # self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 6)
        # self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)

        self.h, self.w = h, w

    def __len__(self):
        return len(self.all_rgbs)


    # def gen_random_rays_at(self, img_idx, batch_size):
    #     pixels_x = torch.randint(low=0, high=self.w, size=[batch_size])
    #     pixels_y = torch.randint(low=0, high=self.h, size=[batch_size])
    #     color = self.all_rgbs[img_idx] # [h, w, 3]
    #     color = color[(pixels_y, pixels_x)]  # [batch_size, 3]
    #     depth = self.all_depth[img_idx].cuda()[(pixels_y, pixels_x)]       # [batch_size,]
    #     depth = depth.unsqueeze(-1).cpu()
    #     mask = torch.ones_like(color, dtype=torch.float)
    #     all_rays = self.all_rays[img_idx].reshape(self.h, self.w, 6) # [h, w, 6]
    #     rand_rays = all_rays[(pixels_y, pixels_x)] # [batch_size, 6]
    #     return torch.cat([rand_rays, color, depth, mask[:, :1]], dim=-1).to(self.device)

    def gen_random_rays_at(self, img_idx, batch_size):
        scene_id = 0
        pixels_x = torch.randint(low=0, high=self.w, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.h, size=[batch_size])
        color = self.all_rgbs[img_idx]    # [h, w, 3]
        color = color[(pixels_y, pixels_x)].cpu()     # [batch_size, 3]
        # depth = self.all_depth[img_idx].cuda()[(pixels_y, pixels_x)]  # [batch_size,]
        # depth = depth.unsqueeze(-1).cpu()
        depth = torch.ones(batch_size, 1).cpu()
        mask = torch.ones_like(color, dtype=torch.float)
        directions = self.directions[scene_id]
        batch_direction = directions[(pixels_y, pixels_x)]
        rays_o, rays_d = get_rays(batch_direction, self.poses[img_idx])  # both (batch_size, 3)
        rand_rays = torch.cat([rays_o, rays_d], -1)
        return torch.cat([rand_rays, color, depth, mask[:, :1]], dim=-1).to(self.device)

    def near_far_from_sphere(self, rays_o, rays_d):
        near = torch.zeros(rays_o.shape[0], 1).cuda()
        far = torch.ones(rays_o.shape[0], 1).cuda() * 3
        return near, far

    def gen_rays_at(self, img_idx, resolution_level=1) -> object:
        tx = torch.linspace(0, self.w-resolution_level, self.w // resolution_level)
        ty = torch.linspace(0, self.h-resolution_level, self.h // resolution_level)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        directions = self.directions[0]
        batch_direction = directions[(pixels_y.long(), pixels_x.long())]
        rays_o, rays_d = get_rays(batch_direction, self.poses[img_idx])  # both (batch_size, 3)
        # scene_idx = self.get_scene_id(img_idx)
        # direction = self.directions[scene_idx]
        # pose = self.poses[img_idx]
        # rays_o, rays_d = get_rays(direction, pose)
        rays_o = rays_o.reshape(self.w//resolution_level, self.h//resolution_level, 3).to(self.device).transpose(0,1)
        rays_d = rays_d.reshape(self.w//resolution_level, self.h//resolution_level, 3).to(self.device).transpose(0,1)
        return rays_o, rays_d

    def image_at(self, idx, resolution_level):
        img = self.all_rgbs[idx].cpu().numpy() * 255
        img = cv2.resize(img, (self.w//resolution_level, self.h//resolution_level))
        return img[:,:,[2,1,0]]

    def get_scene_id(self, img_idx):
        return 0

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        # only used in novel view synthesis
        raise NotImplementedError()



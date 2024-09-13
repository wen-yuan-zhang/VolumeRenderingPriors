import point_cloud_utils as pcu
import trimesh
import numpy as np
import torch
import mcubes
import torch.nn.functional as F
import os

if not os.path.exists('model_watertight.obj'):
    obj_mesh = trimesh.load('model.obj')
    if isinstance(obj_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in obj_mesh.geometry.values()))
    else:
        assert (isinstance(obj_mesh, trimesh.Trimesh))
        mesh = obj_mesh

    v, f = mesh.vertices, mesh.faces
    v *= 2.5
    v, f = pcu.make_mesh_watertight(v,f,resolution=160000)
    mesh = trimesh.Trimesh(v,f)
    mesh.export('model_watertight.obj')
else:
    mesh = trimesh.load('model_watertight.obj')
    v, f = mesh.vertices, mesh.faces


N = 256
resolution = 1024
bound_scale = 2
bound_min = np.array([-bound_scale,-bound_scale,-bound_scale])
bound_max = np.array([bound_scale,bound_scale,bound_scale])
X = torch.linspace(bound_min[0], bound_max[0], resolution, dtype=torch.float32).split(N)
Y = torch.linspace(bound_min[1], bound_max[1], resolution, dtype=torch.float32).split(N)
Z = torch.linspace(bound_min[2], bound_max[2], resolution, dtype=torch.float32).split(N)
u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
with torch.no_grad():
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                pts = pts.numpy().astype(np.float)
                v = np.array(v, dtype=np.float)
                val, fid, bc = pcu.signed_distance_to_mesh(pts, v, f)
                val = val.reshape(len(xs), len(ys), len(zs))
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
                print(xi, yi, zi)

vertices, triangles = mcubes.marching_cubes(u, 0.)
b_max_np = bound_max
b_min_np = bound_min
vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
new_mesh = trimesh.Trimesh(vertices, triangles)
new_mesh.export('model_pcu_'+str(resolution)+'.ply')
np.save('model_volume_'+str(resolution), u)


'''
# test the produced sdf volume
u = np.load('model_volume_'+str(resolution)+'.npy')
u = torch.tensor(u)[None, None, ...]        # [1,1,res,res,res]
bound_min = np.array([-bound_scale,-bound_scale,-bound_scale])
bound_max = np.array([bound_scale,bound_scale,bound_scale])
X = torch.linspace(bound_min[0], bound_max[0], resolution, dtype=torch.float32).split(N)
Y = torch.linspace(bound_min[1], bound_max[1], resolution, dtype=torch.float32).split(N)
Z = torch.linspace(bound_min[2], bound_max[2], resolution, dtype=torch.float32).split(N)

interp = np.zeros([resolution, resolution, resolution], dtype=np.float32)
with torch.no_grad():
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                pts = torch.stack([xx,yy,zz],-1)[None,...]      # [1,res,res,res,3]
                pts = pts[..., [1, 2, 0]] / bound_scale
                pts[..., 0] *= -1
                val = F.grid_sample(u, pts,padding_mode='border', align_corners=True)
                val = val.reshape(len(xs), len(ys), len(zs))
                # val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                interp[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val.numpy()
                print(xi, yi, zi)

vertices, triangles = mcubes.marching_cubes(interp, 0.)
b_max_np = bound_max
b_min_np = bound_min
vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
new_mesh = trimesh.Trimesh(vertices, triangles)
new_mesh.export('model_test_'+str(resolution)+'.ply')
'''
import argparse
import json
import torch
import trimesh
from pyhocon import ConfigFactory
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterizer
from pytorch3d.renderer.cameras import PerspectiveCameras
import trimesh
from scipy.spatial import cKDTree
import numpy as np
import open3d as o3d
import argparse, os, sys
sys.path.append("../")
from datasets.replica import ReplicaDataset


H = 680
W = 1200



translation_dict = {
    # Replica
    'office0': [-0.1944, 0.6488, -0.3271],
    'office1': [-0.585, -0.4703, -0.3507],
    'office2': [0.1909, -1.2262, -0.1574],
    'office3': [0.7893, 1.3371, -0.3305],
    'office4': [-2.0684, -0.9268, -0.1993],
    'room0': [-3.00, -1.1631, 0.1235],
    'room1': [2.0795, 0.1747, 0.0314],
    'room2': [-2.5681, 0.7727, 1.1110],
}

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

scene_bounds_dict = {
    # Replica
    'office0': np.array([[-2.0056, -3.1537, -1.1689],
                             [2.3944, 1.8561, 1.8230]]),
    'office1': np.array([[-1.8204, -1.5824, -1.0477],
                             [2.9904, 2.5231, 1.7491]]),
    'office2': np.array([[-3.4272, -2.8455, -1.2265],
                             [3.0453, 5.2980, 1.5414]]),
    'office3': np.array([[-5.1116, -5.9395, -1.2207],
                             [3.5329, 3.2652, 1.8816]]),
    'office4': np.array([[-1.2047, -2.3258, -1.2093],
                             [5.3415, 4.1794, 1.6078]]),
    'room0': np.array([[-0.8794, -1.1860, -1.5274],
                             [6.8852, 3.5123, 1.2804]]),
    'room1': np.array([[-5.4027, -3.0385, -1.4080],
                             [1.2436, 2.6891, 1.3452]]),
    'room2': np.array([[-0.8171, -3.2454, -2.9081],
                             [5.9533, 1.7000, 0.6861]]),
}



def clean_invisible_vertices(mesh, train_dataset):

    poses = train_dataset.poses
    n_imgs = train_dataset.__len__()
    pc = mesh.vertices
    faces = mesh.faces
    xyz = torch.Tensor(pc)
    xyz = xyz.reshape(1, -1, 3)
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)

    # delete mesh vertices that are not inside any camera's viewing frustum
    whole_mask = np.ones(pc.shape[0]).astype(np.bool)
    for i in tqdm(range(0, n_imgs, 1), desc='clean_vertices'):
        intrinsics = train_dataset.intrinsics
        pose = poses[i]
        camera_pos = torch.einsum('abj,ij->abi', xyz_h, pose.inverse())
        projections = torch.einsum('ij, abj->abi', intrinsics, camera_pos[..., :3])  # [W, H, 3]
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8) - 0.5
        pixel_locations = pixel_locations[:, :, [1, 0]]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        uv = pixel_locations.reshape(-1, 2)
        z = pixel_locations[..., -1:] + 1e-5
        z = z.reshape(-1)
        edge = 0
        mask = (0 <= z) & (uv[:, 0] < H - edge) & (uv[:, 0] > edge) & (uv[:, 1] < W-edge) & (uv[:, 1] > edge)
        whole_mask &= ~mask.cpu().numpy()

    pc = mesh.vertices
    faces = mesh.faces
    face_mask = whole_mask[mesh.faces].all(axis=1)
    mesh.update_faces(~face_mask)

    return mesh

# correction from pytorch3d (v0.5.0)
def corrected_cameras_from_opencv_projection( R, tvec, camera_matrix, image_size):
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Get the PyTorch3D focal length and principal point.
    s = (image_size_wh).min(dim=1).values

    focal_pytorch3d = focal_length / (0.5 * s)
    p0_pytorch3d = -(principal_point - image_size_wh / 2) * 2 / s

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().permute(0, 2, 1)
    # R_pytorch3d = R.clone()
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1
    # T_pytorch3d[:, 0] *= -1

    return PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
    )


def clean_triangle_faces(mesh, train_dataset):
    # returns a mask of triangles that reprojects on at least nb_visible images
    num_view = train_dataset.__len__()
    K = train_dataset.intrinsics[:3, :3].unsqueeze(0).repeat([num_view, 1, 1])
    # keep same with neuralwarp
    R = train_dataset.poses[:, :3, :3].transpose(2, 1)
    t = - train_dataset.poses[:, :3, :3].transpose(2, 1) @ train_dataset.poses[:, :3, 3:]
    sizes = torch.Tensor([[train_dataset.w, train_dataset.h]]).repeat([num_view, 1])
    cams = [K, R, t, sizes]
    num_faces = len(mesh.faces)
    nb_visible = 1
    count = torch.zeros(num_faces, device="cuda")
    K, R, t, sizes = cams[:4]

    n = len(K)
    with torch.no_grad():
        for i in tqdm(range(n), desc="clean_faces"):
            intr = torch.zeros(1, 4, 4).cuda()  #
            intr[:, :3, :3] = K[i:i + 1]
            intr[:, 3, 3] = 1
            vertices = torch.from_numpy(mesh.vertices).cuda().float()  #
            faces = torch.from_numpy(mesh.faces).cuda().long()  #
            meshes = Meshes(verts=[vertices],
                            faces=[faces])

            cam = corrected_cameras_from_opencv_projection(camera_matrix=intr, R=R[i:i + 1].cuda(),  #
                                                           tvec=t[i:i + 1].squeeze(2).cuda(),  #
                                                           image_size=sizes[i:i + 1, [1, 0]].cuda())  #
            cam = cam.cuda()  #
            raster_settings = rasterizer.RasterizationSettings(image_size=tuple(sizes[i, [1, 0]].long().tolist()),
                                                               faces_per_pixel=1)
            meshRasterizer = rasterizer.MeshRasterizer(cam, raster_settings)

            with torch.no_grad():
                ret = meshRasterizer(meshes)
                pix_to_face = ret.pix_to_face
                # pix_to_face, zbuf, bar, pixd =

            visible_faces = pix_to_face.view(-1).unique()
            count[visible_faces[visible_faces > -1]] += 1

    pred_visible_mask = (count >= nb_visible).cpu()

    mesh.update_faces(pred_visible_mask)
    return mesh

def cull_by_bounds(points, scene_bounds):
    eps = 0.2
    inside_mask = np.all(points >= (scene_bounds[0] - eps), axis=1) & np.all(points <= (scene_bounds[1] + eps), axis=1)
    return inside_mask



def crop_mesh(scene, mesh, subdivide=True, max_edge=0.015):
    vertices = mesh.vertices
    triangles = mesh.faces

    if subdivide:
        vertices, triangles = trimesh.remesh.subdivide_to_size(vertices, triangles, max_edge=max_edge, max_iter=10)

    # Cull with the bounding box first
    inside_mask = None
    scene_bounds = scene_bounds_dict[scene]
    if scene_bounds is not None:
        inside_mask = cull_by_bounds(vertices, scene_bounds)

    inside_mask = inside_mask[triangles[:, 0]] | inside_mask[triangles[:, 1]] | inside_mask[triangles[:, 2]]
    triangles = triangles[inside_mask, :]
    print("Processed culling by bound")
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    # we don't need subdivided mesh to render depth
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def transform_mesh(scene, mesh):
    pc = mesh.vertices
    faces = mesh.faces
    pc = (pc / scale_dict[scene]) - np.array([translation_dict[scene]])
    mesh = trimesh.Trimesh(pc, faces, process=False)
    return mesh


def detransform_mesh(scene, mesh):
    pc = mesh.vertices
    faces = mesh.faces
    pc = (pc + np.array([translation_dict[scene]])) * scale_dict[scene]
    mesh = trimesh.Trimesh(pc, faces, process=False)
    return mesh


def monoMesh_2_neusMesh(mesh, scene_name='room0'):
    vertices, faces = mesh.vertices, mesh.faces
    vertices = (vertices + np.array([translation_dict[scene_name]])) * scale_dict[scene_name]
    vertices = (vertices * np.array([[1, -1, 1]]))[:, [0, 2, 1]]
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    return mesh

def compute_iou(mesh_pred, mesh_target):
    res = 0.05
    v_pred = mesh_pred.voxelized(pitch=res)
    v_target = mesh_target.voxelized(pitch=res)
    v_target_mesh = v_target.as_boxes()
    v_pred_mesh = v_pred.as_boxes()

    v_pred_filled = set(tuple(np.round(x, 4)) for x in v_pred.points)
    v_target_filled = set(tuple(np.round(x, 4)) for x in v_target.points)
    inter = v_pred_filled.intersection(v_target_filled)
    union = v_pred_filled.union(v_target_filled)
    iou = len(inter) / len(union)
    return iou, v_target_mesh, v_pred_mesh


def get_colored_pcd(pcd, metric):
    cmap = plt.cm.get_cmap("jet")
    color = cmap(metric / 0.10)[..., :3]
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d


def compute_metrics(mesh_pred, mesh_target):
    # mesh_pred = trimesh.load_mesh(path_pred)
    # mesh_target = trimesh.load_mesh(path_target)
    area_pred = int(mesh_pred.area * 1e4)
    area_tgt = int(mesh_target.area * 1e4)
    print("pred: {}, target: {}".format(area_pred, area_tgt))

    iou, v_gt, v_pred = compute_iou(mesh_pred, mesh_target)

    pointcloud_pred, idx = mesh_pred.sample(area_pred, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx]

    pointcloud_tgt, idx = mesh_target.sample(area_tgt, return_index=True)
    pointcloud_tgt = pointcloud_tgt.astype(np.float32)
    normals_tgt = mesh_target.face_normals[idx]

    thresholds = np.array([0.05])

    # for every point in gt compute the min distance to points in pred
    completeness, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud_pred, normals_pred
    )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness ** 2

    # color gt_point_cloud using completion
    # com_mesh = get_colored_pcd(pointcloud_tgt, completeness)

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, normals_pred, pointcloud_tgt, normals_tgt
    )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy ** 2

    # color pred_point_cloud using completion
    # acc_mesh = get_colored_pcd(pointcloud_pred, accuracy)

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
    )
    chamferL1 = 0.5 * (completeness + accuracy)

    # F-Score
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]
    rst = {
        "IoU": iou,
        # "Acc": accuracy,
        # "Comp": completeness,
        "C-L1": chamferL1,
        "NC": normals_correctness,
        'precision': precision[0],
        'recall': recall[0],
        "F-score": F[0]
    }

    return rst


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """ Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    """ Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [
        (dist <= t).astype(np.float32).mean() for t in thresholds
    ]
    return in_threshold



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/Replica/')
    parser.add_argument('--gt_dir', type=str, default='../data/Replica/gt_meshes/culled_meshes/')
    parser.add_argument('--mesh_dir', type=str, default='../log/replica/room0/')
    parser.add_argument('--mesh_name', type=str, default='udf_res512_step300000.ply')
    parser.add_argument('--scan', type=str, default='room0')
    parser.add_argument('--vis_out_dir', type=str, default='.')
    parser.add_argument('--downsample_density', type=float, default=0.002)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=0.1)
    parser.add_argument('--visualize_threshold', type=float, default=0.01)
    parser.add_argument('--log', type=str, default=None)
    args = parser.parse_args()


    scene = args.scan
    print('processing', scene)
    exp_dir = args.mesh_dir
    mesh_name = args.mesh_name
    exp_mesh_name = os.path.join(exp_dir, mesh_name)
    gt_mesh_dir = os.path.join(args.gt_dir,str(scene)+'.ply')
    conf = ConfigFactory.parse_string("dataset { \n"
                                      "data_dir=%s \n"
                                      "scene=%s \n"
                                      "}"
                                      % (args.data_dir, scene))

    train_dataset = ReplicaDataset(conf['dataset'])

    mesh = trimesh.load_mesh(exp_mesh_name)

    ### CULL
    cull_mesh_name = mesh_name[:-4]+'_culled.ply'
    if not os.path.exists(os.path.join(exp_dir, 'results', cull_mesh_name)):
        mesh.vertices = mesh.vertices[:, [0,2,1]] * np.array([[1, -1, 1]])
        mesh = transform_mesh(scene, mesh)
        mesh = crop_mesh(scene, mesh)
        mesh = detransform_mesh(scene, mesh)
        mesh.vertices = (mesh.vertices * np.array([[1, -1, 1]]))[:, [0, 2, 1]]
        mesh = clean_invisible_vertices(mesh, train_dataset)
        mesh = clean_triangle_faces(mesh, train_dataset)
        mesh.vertices = mesh.vertices[:, [0, 2, 1]] * np.array([[1, -1, 1]])
        mesh = transform_mesh(scene, mesh)
        os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
        mesh.export(os.path.join(exp_dir, 'results', mesh_name[:-4]+'_culled.ply'))
    else:
        mesh = trimesh.load(os.path.join(exp_dir, 'results', cull_mesh_name))


    ### EVAL
    gt_mesh = trimesh.load_mesh(gt_mesh_dir)

    ret = compute_metrics(mesh, gt_mesh)
    print(ret)
    for key in ret.keys():  # for json dump
        ret[key] = float(ret[key])
    with open(os.path.join(exp_dir, 'results', f'results_{mesh_name[:-4]}.json'), 'w') as f:
        os.makedirs(os.path.join(exp_dir,'results'),exist_ok=True)
        json.dump(ret, f, indent=2)
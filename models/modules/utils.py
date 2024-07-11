import numpy as np

import torch
from torch import nn, Tensor

import torch_geometric.nn as geo_nn
from torch_geometric.nn import knn

from dgl.geometry import farthest_point_sampler

import gc

activation_funcs = ['relu', 'leaky_relu', 'elu', 'gelu']

def get_activation_func(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        raise Exception('The input activation function {} is not defined.'.format(activation)) 

def points_sampling(sample_method: str, nsample: int, points: Tensor, features: Tensor):
    '''
    This function returns the sampled points of point clouds based on give
    method name.
    '''
    assert sample_method in ['rs', 'fsp']    # rs means random sampling, fsp means farthest point sampling
    assert len(points.shape) == 3            # B, N, 3

    B, N, pdim = points.shape
    _, _, fdim = features.shape   
    device = features.device
    xyzs = points[..., 0:3]
    if sample_method == 'rs':   
        selected_idx = torch.multinomial(torch.ones([B, N]), 
                                         num_samples=nsample, replacement=False) # (B, sampled_N)
    elif sample_method == 'fsp':
        selected_idx = farthest_point_sampler(xyzs, nsample)                      # (B, sampled_N)

    selected_idx = torch.unsqueeze(selected_idx, dim=-1)
    point_selector = selected_idx.repeat(1, 1, pdim).to(device)
    feature_selector = selected_idx.repeat(1, 1, fdim).to(device)

    selected_points = torch.gather(points, 1, point_selector)   # .to(device)
    selected_feature = torch.gather(features, 1, feature_selector)   # .to(device)  
    del point_selector, feature_selector
    return selected_idx, selected_points, selected_feature


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data

def rotate_point_cloud_z(batch_data, angle=360):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    B, _, _ = batch_data.shape
    device = batch_data.device

    angle = angle/180*np.pi
    for k in range(B):
        rotation_angle = torch.rand(1)*angle
        cosval = torch.cos(rotation_angle)
        sinval = torch.sin(rotation_angle)
        rotation_matrix = torch.tensor([[cosval, sinval, 0],
                                        [-sinval, cosval, 0],
                                        [0, 0, 1]]).to(device)
        shape_pc = batch_data[k, ...]
        batch_data[k, ...] = torch.matmul(shape_pc.reshape((-1, 3)), rotation_matrix)
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, _, _ = batch_data.shape
    device = batch_data.device
    shifts = (shift_range*2)*torch.rand(B, 3)-shift_range
    shifts = shifts.to(device)
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = (scale_high-scale_low)*torch.rand(B,)+scale_low
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data

def index_points(feature, grouped_idx):
    """
    Given points and corresponded idx for neighbour, return the neighbour points and their features.
    Input:
        points: input points data, [B, N_p, D]
        grouped_id: sample index data, [B, N_g, npoint]
    Return:
        new_points:, indexed points data, [B, N, npoint, D]
    """
    B, _, D = feature.shape
    _, N_g, npoint = grouped_idx.shape

    with torch.no_grad():
        grouped_idx = torch.unsqueeze(grouped_idx.reshape(B, -1), dim=-1).expand(B, N_g*npoint, D)
    
    new_feature = torch.gather(feature, 1, grouped_idx).reshape(B, N_g, npoint, D)
    del grouped_idx
    torch.cuda.empty_cache()
    return new_feature

def index_sampling(xyz, feature, idx_global, idx_local):
    '''
    Sampling the points with respect to its center points
    Args:
        xyz: [B, 3, N, 1]
        feature: [B, M, C]
        idx_global: [B, sampled_N]
        idx_local: [B, sampled_N]
    Returns:
        pool_xyz: [B, sampled_N, 3]
        pool_features: [B, sampled_N, C]
    '''

    B, N, C = feature.shape

    selected_xyz_idx = torch.unsqueeze(idx_global, dim=-1).expand(B, idx_global.shape[-1], 3)          # (B, sampled_N, 3)
    selected_feature_idx = torch.unsqueeze(idx_local, dim=-1).expand(B, idx_local.shape[-1], C)        # (B, sampled_N, C)

    feature = torch.gather(feature, 1, selected_feature_idx)                                               # (B, sampled_N, C)
    xyz = torch.gather(xyz, 1, selected_xyz_idx)
    return xyz, feature

def dynamic_slice(x, starts, sizes):
    # starts = (0..., chun_idx, 0, 0)
    # sizes = (size_0..., key_chunk_size, num_heads, k_dim)
    # only to adjust the key_chunk_size to prevent it from going above N size
    starts = [np.clip(starts[i], 0, x.shape[i] - sizes[i]) for i in range(len(starts))] 
    for i, (start, size) in enumerate(zip(starts, sizes)):    # this loop might be redundant
        x = torch.index_select(x, i, torch.tensor(range(start, start + size), device=x.device))
    return x

def map_pt(f, xs):
    t = [f(x) for x in xs]
    return tuple(map(torch.stack, zip(*t)))

def memory_check(logger):
    '''
    Pass the logger and it will return the list of tensor in the device.
    '''
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                logger.info(str(type(obj)) + '    ' + str(obj.size()))
        except: pass

def same_storage(x, y):
    '''
    Check if all elements from two tensors share the same address. 
    '''
    x_ptrs = set(e.data_ptr() for e in x.view(-1))
    y_ptrs = set(e.data_ptr() for e in y.view(-1))
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)


def knn_algorithm(nsample, xyz1, xyz2):
    '''
    This is the knn algorithm implemented based on torch_geometric.   
    Find the element in xyz2 the k nearest poinst in xyz1 
    Input:
        nsample: number of points 
        xyz1: [B, N, dim]
        xyz2: [B, M, dim]
        N > M    
    '''
    B, M, _ = xyz2.shape
    device = xyz2.device
    grouped_idx = torch.zeros([B, M, nsample]).to(device)
    for i in range(B):
        a = knn(xyz1[i, ...], xyz2[i, ...], k=nsample+1)
        a = a[1, ...].reshape(M, nsample+1)
        grouped_idx[i, ...] = a[:, 1:]
    grouped_idx = grouped_idx.to(dtype=torch.long)
    return grouped_idx

def ball_query_algorithm(radius, nsample, xyz1, xyz2):
    '''
    This is the ball algorithm re-implemented based on torch_geometric.   
    Find the element in xyz2 points within radius in xyz1. 
    Input:
        nsample: number of points 
        xyz1: [B, N, dim]
        xyz2: [B, M, dim]
        N > M    
    '''
    B, M, _ = xyz2.shape
    device = xyz2.device

    # _, ball_idx, _ = ball_query(p1=xyz2, p2=xyz1, K=nsample, radius=radius, return_nn=False)
    grouped_idx = []
    for i in range(B):
        ball_idx = geo_nn.radius(x=xyz1[i, ...], y=xyz2[i, ...], 
                                 r=radius, max_num_neighbors=nsample)
        xyz1_idx = ball_idx[1, :]
        xyz2_idx = ball_idx[0, :]
        neighs_idx = []
        for i in range(M):
            neigh_idx_bool = (xyz2_idx==i)
            # no neighbors exsit -> use itself as neigh
            if torch.sum(neigh_idx_bool) == 0:
                neigh_idx = torch.full([nsample], fill_value=i)
            else:
                neigh_idx = xyz1_idx[neigh_idx_bool]
                if neigh_idx.shape[0] < nsample:
                    residual = int(abs(neigh_idx.shape[0]-nsample))
                    choice = np.random.choice(neigh_idx.shape[0], 
                                              residual, replace=True)  # There will be repeated points
                    neigh_idx = torch.concat([neigh_idx, neigh_idx[choice]], dim=0)
            neighs_idx.append(neigh_idx.unsqueeze(0))
        neighs_idx = torch.concat(neighs_idx, dim=0)
        grouped_idx.append(neighs_idx.unsqueeze(dim=0)) 
    grouped_idx = torch.concat(grouped_idx, dim=0)

    return grouped_idx

def geo_feature_encoding(points, grouped_points, nsample):
    '''
    Input:
        points: [B, N, 3 + ?]  --> there could be probably other geometrical features to be added
        grouped_points: [B, N, nsample, 3 + ?]
    Output:
        expanded_fs: [B, N, nsample, n_fs]
        density map: [B, N], which maps the density of each points by add the distances of neighbor points
    '''
    xyz = points[:, :, 0:3]
    grouped_xyz = grouped_points[..., 0:3]
    B, N, nsample, _ = grouped_points.shape

    xyz_expanded = torch.unsqueeze(xyz, dim=2).expand(B, N, nsample, 3)      # [B, N, nsample, 3]
    rela_xyz = xyz_expanded - grouped_xyz                                     # [B, N, nsample, 3]
    rela_dis = torch.linalg.norm(rela_xyz, dim = -1, keepdim=True)            # [B, N, nsample, 1]
    # concatenate                [3,          3,           3,           1]
    expanded_f = torch.cat([xyz_expanded, grouped_points, rela_xyz,  rela_dis], dim=-1)
    return expanded_f


def pairwise_distance(
    x: Tensor, y: Tensor, normalized: bool=False, channel_first: bool = False) -> Tensor:
    r"""Pairwise distance of two (batched) point clouds.
    https://github.com/qinzheng93/GeoTransformer/blob/main/geotransformer/modules/ops/pairwise_distance.py
    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances

def inverse_density_sampling_idx(xyz, feature, idx_global, idx_local):
    '''
    Sampling the points with respect to its center points
    Args:
        xyz: [B, 3, N, 1]
        feature: [B, M, C]
        idx_global: [B, sampled_N]
        idx_local: [B, sampled_N]
    Returns:
        pool_xyz: [B, sampled_N, 3]
        pool_features: [B, sampled_N, C]
    '''
    
    B, N, C = feature.shape
    _, _, geodim = xyz.shape
    selected_xyz_idx = torch.unsqueeze(idx_global, dim=-1).expand(B, idx_global.shape[-1], geodim)         # (B, sampled_N, 3)
    selected_feature_idx = torch.unsqueeze(idx_local, dim=-1).expand(B, idx_local.shape[-1], C)            # (B, sampled_N, C)

    feature = torch.gather(feature, 1, selected_feature_idx)                                               # (B, sampled_N, C)
    xyz = torch.gather(xyz, 1, selected_xyz_idx)
    return xyz, feature

def cls_avg_acc(predict, target, num_part):
    '''
    Return the acc for each object in one batch
    Args:
        predict: predicted label (B, N)
        target: true label (B, N)

    Returns:
        cls_avg_acc: average accuracy for each class (num_cls,)
        exist_flag: (num_cls,) 0 if the class does not exist
    '''
    B, N = predict.shape
    correct_p_cls = np.zeros(5)
    total_p_cls = np.zeros(5)
    exist_flag = np.ones(5)
    for l in range(num_part):
        total_p_cls[l] = np.sum(target == l)                                             # True label for each class
        if total_p_cls[l] == 0:
            exist_flag[l] = 0
            continue
        correct_p_cls[l] = (np.sum((predict == l) & (target == l)))       # for the component been rightly classified
    #avg_acc = np.array([correct_p_cls[i]/total_p_cls[i] if total_p_cls[i] != 0 else 0 for i in range(num_part)])
    return correct_p_cls, total_p_cls, exist_flag


def cls_avg_iou(predict, target, num_part):
    '''
    Return the acc for each object in one batch
    Args:
        predict: predicted label (B, N)
        target: true label (B, N)

    Returns:
        cls_avg_acc: average accuracy for each class (num_cls,)
        exist_flag: (num_cls,) 0 if the class does not exist
    '''
    if len(predict.shape) == 1:
        predict = np.expand_dims(predict, axis=0)
    if len(target.shape) == 1:
        target = np.expand_dims(target, axis=0)
    B, N = predict.shape
    total_iou = np.zeros(5)
    p_cls_num = np.zeros(5)
    for i in range(B):
        batch_predict = predict[i, :]                                                              # predicted
        batch_target = target[i, :]                                                                # true
        for l in range(num_part):
            if (np.sum(batch_predict == l) == 0) and (np.sum(batch_target == l) == 0):             # part is not present, no prediction as well
                continue
            else:
                part_iou = np.sum((batch_predict == l) & (batch_target == l)) / float(np.sum((batch_predict == l) | (batch_target == l)))
                total_iou[l] += part_iou
                p_cls_num[l] += 1
    avg_iou = np.array([total_iou[i]/p_cls_num[i] if p_cls_num[i] != 0 else 0 for i in range(num_part)])
    return avg_iou

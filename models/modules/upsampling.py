import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import knn_interpolate

from models.modules.utils import index_points

'''Bridge upsampling layers'''
class UpInterpoaltion_knn(nn.Module):
    '''
    This is the upsampling layer which use the interpolation introduced from PointNet++.
    '''
    def __init__(self, in_ch:int, mlp:list[int], norm_before, activation_func, k:int=3):
        super(UpInterpoaltion_knn, self).__init__()
        self.k = k
        self.norm_before = norm_before
        self.activation_func = activation_func
        self.mlp_layers = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_ch = in_ch
        for out_ch in mlp:
            self.mlp_layers.append(nn.Linear(last_ch, out_ch))
            last_ch = out_ch
        if self.norm_before:
            self.norm = (nn.LayerNorm(in_ch))
        else:
            self.norm = (nn.LayerNorm(out_ch))
        

    def forward(self, feature_1:Tensor, feature_2:Tensor, 
                      points_1:Tensor, points_2:Tensor):
        """
        Input:
            feature_1: input points data, [B, S, in_ch]
            feature_2: points from sub-sampling layers, , [B, N, in_ch]
            points_1: input points position data, [B, S, 3+3] --> N is the number of the point
            points_2: sampled input points position data, [B, N, 3+3]  --> S is the number of the point, which S < N
            k: k nearest points
        Return:
            new_points: upsampled points data, [B, N, out_ch]
        """
        
        device = feature_1.device
        B, _, C = feature_1.shape
        _, N, _ = points_2.shape       # first time it will equal to 1, so the 'if' is activated
        
        xyz1 = points_1[..., 0:3]
        xyz2 = points_2[..., 0:3]
        new_features = torch.zeros([B, N, C]).to(device)
        for i in range(B):
            new_features[i, ...] = knn_interpolate(feature_1[i, ...], xyz1[i, ...], xyz2[i, ...], k=self.k)
        concat_f = torch.cat([new_features, feature_2], dim=-1)
        if self.norm_before:
            feature = self.pre_ln(concat_f)
        else:
            feature = self.post_ln(concat_f)
        
        return feature

    def post_ln(self, feature):
        for _, mlp_layer in enumerate(self.mlp_layers):
            feature = self.activation_func(mlp_layer(feature))
        feature = self.norm(feature)
        return feature
    
    def pre_ln(self, feature):
        feature = self.norm(feature)
        for _, mlp_layer in enumerate(self.mlp_layers):
            feature = self.activation_func(mlp_layer(feature))
        return feature
    
class BridgeNetUp_knn(nn.Module):
    def __init__(self, in_ch, mlp, k=3):
        super(BridgeNetUp_knn, self).__init__()
        self.k = k
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_ch
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, points1, points2, xyz1, xyz2):
        """
        Input:
            points1: input points data, [B, S, C]  --> S is the number of the point, which S < N
            points2: points from sub-sampling layers  [B, N, C]
            grouped_idx: [B, N, nsample]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # print('feature_1: {}, feature_2: {}'.format(points1.shape, points2.shape))
        device = points1.device
        B, S, C = points1.shape
        _, N, _ = points2.shape       # first time it will equal to 1, so the 'if' is activated
   
        new_features = torch.zeros([B, N, C]).to(device)
        for i in range(B):
            new_features[i, ...] = knn_interpolate(points1[i, ...], xyz1[i, ...], xyz2[i, ...], k=self.k)      
        concat_f = torch.cat([new_features, points2], dim=-1)
        concat_f = concat_f.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            concat_f = F.relu(bn(conv(concat_f)))
        concat_f = concat_f.permute(0, 2, 1)
        return concat_f
    
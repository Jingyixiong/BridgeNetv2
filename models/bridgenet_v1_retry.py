'''
Re-define the bridgenetv1 to fit bridgenetv2's training framework
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn

from models.modules.upsampling import BridgeNetUp_knn
from models.modules.utils import index_points, points_sampling, ball_query_algorithm

class BridgeNet(nn.Module):   # inhererit from nn.Module
    def __init__(self, in_ch, n_cls, nsample_dict, block_n_dict, n_feature,
                 in_ch_dict, out_ch_dict, m_dict, radius_dict, 
                 sampling_method: str, sampling_ratio: list[int],
                 neigh_mode='ball_query', downsample='maxpooling'):

        super(BridgeNet, self).__init__()
        self.downsample = downsample
        self.sampling_method = sampling_method
        self.sampling_ratio = sampling_ratio

        self.conv_1 = nn.Conv2d(in_ch, 8, 1)
        self.bn_1   = nn.BatchNorm2d(8)
        self.bnm_list = nn.ModuleList()
        self.radius_dict = radius_dict

        for i in range(len(block_n_dict)):      # there are default 4 layers
            name = 'bnm_{}'.format(i + 1)
            nsample = nsample_dict[name]
            n_block = block_n_dict[name]
            in_ch = in_ch_dict[name]
            out_ch = out_ch_dict[name]
            m = m_dict[name]
            radius = self.radius_dict[name]
            bnm = BridgeNet_Block(
                in_ch=in_ch, out_ch=out_ch, 
                n_feature=n_feature,
                n_block=n_block, m=m, 
                neigh_mode=neigh_mode, nsample=nsample, radius=radius,
                downsample=self.downsample
                )
            self.bnm_list.append(bnm)

        self.global_fea = Global_Fea_Extractor(out_ch[-1])

        self.conv_2 = nn.Conv2d(out_ch[-1], out_ch[-1], 1)
        self.bn_2   = nn.BatchNorm2d(out_ch[-1])

        self.fp4 = BridgeNetUp_knn(512, [256, 128], k=3)
        self.fp3 = BridgeNetUp_knn(256, [128, 64], k=3)
        self.fp2 = BridgeNetUp_knn(128, [64, 32], k=3)
        self.fp1 = BridgeNetUp_knn(64, [64, 32], k=3)

        self.conv_fc1 = nn.Conv1d(32, 64, 1)     # only accept data with [B, D, N]
        self.bn_fc1  = nn.BatchNorm1d(64)
        self.conv_fc2 = nn.Conv1d(64, 32, 1)     # only accept data with [B, D, N]
        self.bn_fc2  = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.8)

        self.conv_out = nn.Conv1d(32, n_cls, 1)
        self.bn_out   = nn.BatchNorm1d(n_cls)

    def forward(self, points): 
        # Encoding Layer
        points = points[..., 0:3]
        points = torch.unsqueeze(points, dim=-1).permute(0, 2, 1, 3)
    
        points_list, feature_encode_list, feature = self.bridgenet(points,)

        feature = self.global_fea(feature)

        feature = torch.unsqueeze(feature, dim=-1).permute(0, 2, 1, 3)
        feature = F.relu(self.bn_2(self.conv_2(feature)))
        feature = torch.squeeze(feature).permute(0, 2, 1)

        feature = self.fp4(feature, feature_encode_list[-1], points_list[-1], points_list[-2])
        feature = self.fp3(feature, feature_encode_list[-2], points_list[-2], points_list[-3])
        feature = self.fp2(feature, feature_encode_list[-3], points_list[-3], points_list[-4])
        feature = self.fp1(feature, feature_encode_list[-4], points_list[-4], points_list[-5])
        
        feature = feature.permute(0, 2, 1)
        feature = F.relu(self.bn_fc1(self.conv_fc1(feature)))
        feature = F.relu(self.bn_fc2(self.conv_fc2(feature)))
        feature = self.drop1(feature)
        feature = self.bn_out(self.conv_out(feature))
        feature = F.log_softmax(feature, dim=1)
        feature = feature.permute(0, 2, 1)

        return feature

    def bridgenet(self, points):
        _, _, N, _ = points.shape
        n_points_sampled = [int(N/i) for i in self.sampling_ratio]          # the number of point be sampled 

        feature_encode_list = []
        points_list = []
        feature = F.relu(self.bn_1(self.conv_1(points)))
        points, feature = torch.squeeze(points).permute(0, 2, 1), torch.squeeze(feature).permute(0, 2, 1)
        points_list.append(points)
        # sub_sampling process
        sampled_points = torch.zeros_like(points)
        for i in range(len(self.sampling_ratio)):
            bnm = self.bnm_list[i]
            if i == 0:
                feature = bnm(points, feature)
                sampled_points = points
            else:
                feature = bnm(sampled_points, feature)

            feature_encode_list.append(feature)
            _, sampled_points, feature = points_sampling(self.sampling_method, n_points_sampled[i],
                                                         points=sampled_points, features=feature)
            points_list.append(sampled_points)

        return points_list, feature_encode_list, feature


class BridgeNet_Block(nn.Module):
    def __init__(self, in_ch, out_ch, n_feature, n_block, m, neigh_mode, nsample:int, 
                 radius: float=0, downsample='maxpooling'):
        '''
        Args:
            radius_list:
            nsample_list:
            in_ch: [], list of int indicates the input channel of the feature
            out_ch: [], list of int indicates the output channel of the feature
            n_feature: Number of features directly from (x, y, z) coordinates [xi, xj, |xi - xj|, ||xi - xj||]
            geo_fea_up: if True, it first make the last dim of the geo-features the same as the input feature tensor
        '''
        super(BridgeNet_Block, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.n_block = n_block
        self.n_feature = n_feature
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.m = m
        self.neigh_mode = neigh_mode
        assert self.neigh_mode in ['knn', 'ball_query']

        self.sample_ch = int(self.out_ch[-1]/self.m)
        assert self.sample_ch > 1

        self.downsample = downsample

        # Geometric feature Convolutional layer
        self.conv_geo_list = nn.ModuleList()
        self.bn_geo_list = nn.ModuleList()
        in_geo_ch = self.n_feature

        self.sub_conv_geo = nn.Conv2d(in_geo_ch, self.sample_ch, 1)
        self.sub_bn_geo = nn.BatchNorm2d(self.sample_ch)

        self.lme_list = nn.ModuleList()
        self.conv_res_list = nn.ModuleList()
        self.bn_res_list = nn.ModuleList()
        for i in range(self.n_block):          # only two lme layers are added
            lme = LocFeacoding(nsample, self.in_ch[i], self.out_ch[i], self.sample_ch,
                               self.n_feature, self.radius, downsample=self.downsample)
            self.lme_list.append(lme)

    def forward(self, points, features):
        """
        Input:
            points: input points position data, [B, N, 3+?]
            features: input points data, [B, N, ch_in]
            grouped_id: supported grouped_id from pre-computation [B, N, 64 or 128]
        Return:
            points: sample points feature data, [B, N, ch_out]
        """
        if self.neigh_mode == 'knn':
            grouped_idx = self.knn_algorithm(self.nsample, points)
        elif self.neigh_mode == 'ball_query':
            assert self.radius > 0
            grouped_idx = ball_query_algorithm(self.radius, self.nsample, points, points)
        assert len(grouped_idx.shape) == 3
        grouped_points = index_points(points, grouped_idx)                                  # [B, N, nsample, 3 + ?]
        expanded_f = self.geo_feature_encoding(points, grouped_points, self.nsample)        # [B, N, nsample, 10]
        expanded_f = expanded_f.permute(0, 3, 1, 2)
        expanded_f = F.relu(self.sub_bn_geo(self.sub_conv_geo(expanded_f)))
        features = torch.unsqueeze(features.permute(0, 2, 1), dim=-1)

        for i in range(len(self.lme_list)):
            features = self.lme_list[i](expanded_f, features, grouped_idx)           # [B, 4*d_in, N, 1], _

        features = torch.squeeze(features, dim=-1).permute(0, 2, 1)
        return features

    @staticmethod
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
        B, N, _ = points.shape

        xyz_expanded = torch.unsqueeze(xyz, dim=2).expand(B, N, nsample, 3)      # [B, N, nsample, 3]
        rela_xyz = xyz_expanded - grouped_xyz                                     # [B, N, nsample, 3]
        rela_dis = torch.linalg.norm(rela_xyz, dim = -1, keepdim=True)            # [B, N, nsample, 1]
        # concatenate                [3,          3,           3,           1]
        expanded_f = torch.cat([xyz_expanded, grouped_points, rela_xyz,  rela_dis], dim=-1)
        return expanded_f

    @staticmethod
    def knn_algorithm(nsample, xyz):
        '''
        This is the knn algorithm implemented based on torch_geometric.        
        '''
        B, N, _ = xyz.shape
        device = xyz.device
        grouped_idx = torch.zeros([B, N, nsample]).to(device)
        for i in range(B):
            a = knn(xyz[i, ...], xyz[i, ...], k=nsample+1)
            a = a[1, ...].reshape(N, nsample+1)
            grouped_idx[i, ...] = a[:, 1:]
        grouped_idx = grouped_idx.to(dtype=torch.long)
        return grouped_idx
        
class LocFeacoding(nn.Module):
    def __init__(self, nsample, in_ch, out_ch, sample_ch, n_feature, radius, residual=True, downsample='maxpooling'):
        '''
            Encode the geometric feature
        '''
        super(LocFeacoding, self).__init__()
        self.nsample = nsample
        self.n_feature = n_feature
        self.residual = residual
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.sample_ch = sample_ch
        self.radius = radius
        self.downsample = downsample

        self.sub_conv = nn.Conv2d(self.in_ch, self.sample_ch, 1)
        self.sub_bn = nn.BatchNorm2d(self.sample_ch)

        self.up_conv = nn.Conv2d(self.sample_ch, self.out_ch, 1)
        self.up_bn = nn.BatchNorm2d(self.out_ch)

        if (residual == True) or (in_ch != out_ch):      # this is applied when ch_in is not the same as ch_out
            self.conv_res = nn.Conv2d(self.in_ch, self.out_ch, 1)
            self.bn_res = nn.BatchNorm2d(self.out_ch, 1)

        self.downsample_feacoding = DownSample_Feacoding(self.sample_ch, self.radius, self.nsample, self.n_feature,
                                                         self.downsample)

    def forward(self, expanded_f, feature, grouped_idx):
        sub_feature = F.relu(self.sub_bn(self.sub_conv(feature)))               # [B, d_in/m, N, 1]
        sub_feature = torch.squeeze(sub_feature, dim=-1).permute(0, 2, 1)       # [B, N, d_in/m]
        expanded_f = expanded_f.permute(0, 2, 3, 1)

        # This is the module which to process the down-sampled tensors
        concat_f = self.downsample_feacoding(sub_feature, expanded_f, grouped_idx)

        concat_f = F.relu(self.up_bn(self.up_conv(concat_f)))     # up-sample it back to original
        if (self.residual == True) or (self.in_ch != self.out_ch):
            res_fs = F.relu(self.bn_res(self.conv_res(feature)))
            concat_f = concat_f + res_fs
        else:
            concat_f = concat_f + feature

        return concat_f

class DownSample_Feacoding(nn.Module):
    '''
        This module is to process the down-sampled feature vector and the geometric vector.
        The output ch should be same as input ch
    '''
    def __init__(self, sample_ch, radius, nsample, feature_ch, downsample):
        super(DownSample_Feacoding, self).__init__()
        self.sample_ch = sample_ch
        self.group_ch = self.sample_ch*2
        self.radius = radius
        self.nsample = nsample
        self.downsample = downsample
        self.feature_ch = feature_ch

        self.att_conv_fea = nn.Linear(nsample, nsample)     # generate a [nsample, nsample] matrix to make the feature attentive
        self.att_conv_catfea = nn.Linear(nsample, 1)        # generate a [nsmaple, 1] matrix to make the concatenated feature attentive

        self.conv_group_1 = nn.Conv2d(self.group_ch, self.sample_ch, 1)
        self.bn_group_1   = nn.BatchNorm2d(self.sample_ch)

        self.conv_out = nn.Conv2d(self.group_ch, self.sample_ch, 1)
        self.bn_out = nn.BatchNorm2d(self.sample_ch)

    def forward(self, feature, expanded_fs, grouped_idx):
        '''
        Args:
            feature: [B, N, d_in]
            grouped_fea: [B, N, nsample, d_in]
            expanded_fs: [B, N, nsample, d_in]
        Returns:
        '''
        grouped_f = index_points(feature, grouped_idx)
        concat_fea = torch.cat([expanded_fs, grouped_f], dim=-1)      # [B, N, nsample, ch_sample*2]

        # Geometric Convolutional Modeling (GCM)
        concat_fea = concat_fea.permute(0, 3, 1, 2)
        concat_fea = F.relu(self.bn_group_1(self.conv_group_1(concat_fea)))
        concat_fea = concat_fea.permute(0, 2, 3, 1)

        # Attentional Aggregation (AG)
        B, N, nsample, ch = concat_fea.shape
        concat_fea_score = concat_fea.reshape(B*N, nsample, ch).permute(0, 2, 1)         # [B*N, (ch_sample+3)*4, nsample]
        cat_corr_score = self.att_conv_catfea(concat_fea_score).reshape(B, N, ch)        # [B, N, (ch_sample+3)*4]
        cat_corr_score = torch.softmax(cat_corr_score, dim=-1)                           # [B, N, (ch_sample+3)*4]

        if self.downsample == 'sum':
            concat_fea = torch.sum(concat_fea, dim=2, keepdim=True)                      # [B, (ch_sample+3)*4, N]
        elif self.downsample == 'maxpooling':
            concat_fea, _ = torch.max(concat_fea, dim=2, keepdim=True)                   # [B, (ch_sample+3)*4, N]
        elif self.downsample == 'mean':
            concat_fea = torch.mean(concat_fea, dim=2, keepdim=True)

        concat_fea_weighted = concat_fea * torch.unsqueeze(cat_corr_score, dim=2)        # [B, (ch_sample+3)*4, N]
        concat_fea = concat_fea + concat_fea_weighted
        concat_fea = concat_fea.permute(0, 3, 1, 2)

        return concat_fea

'''Bridge middle layer'''
class Global_Fea_Extractor(nn.Module):
    def __init__(self, ch_in):
        super(Global_Fea_Extractor, self).__init__()
        self.ch_in = ch_in

        self.conv_latent_1 = nn.Conv1d(self.ch_in, 1, 1)
        self.bn_latent_1 = nn.BatchNorm1d(1)

        self.conv_latent_2 = nn.Conv1d(self.ch_in, 1, 1)
        self.bn_latent_2 = nn.BatchNorm1d(1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature):
        B, N, _ = feature.shape
        feature = feature.permute(0, 2, 1)                                                          # (B, C, N)
        latent_repre_1 = F.relu(self.bn_latent_1(self.conv_latent_1(feature))).permute(0, 2, 1)     # (B, N, 1)
        latent_repre_2 = F.relu(self.bn_latent_2(self.conv_latent_2(feature)))                      # (B, 1, N)
        att_matrix = torch.matmul(latent_repre_1, latent_repre_2)                                   # (B, N, N)
        att_matrix = self.softmax(att_matrix.reshape(B, -1)).reshape(B, N, N)
        feature = feature.permute(0, 2, 1)                                                          # (B, N, C)
        feature = torch.matmul(att_matrix, feature) + feature                                       # (B, N, C)
        return feature

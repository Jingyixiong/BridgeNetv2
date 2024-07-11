'''
This module contains the feature extractor from the BridgeNetv1
'''
import torch
from torch import nn

from models.modules.postional_encode import GeometricNeigh_Embedding
from models.modules.utils import index_points, geo_feature_encoding

class BridgeNet_FF_v2(nn.Module):
    def __init__(self, in_ch_list, out_ch_list, geo_ch, f_sample_r, nsample:int, 
                 activation_func, extractor_pos_mode, pos_config, block_n=2,
                 downsample='maxpooling'):
        '''
        BridgeNetv1 feature block.
        Args:
            radius_list:
            nsample_list:
            in_ch: [], list of int indicates the input channel of the feature
            out_ch: [], list of int indicates the output channel of the feature
            geo_ch: Number of features directly from (x, y, z) coordinates [xi, xj, |xi - xj|, ||xi - xj||]
            f_sample_r: sampling ratio of features
            normal_flag: whether to use the normal vector information in rotinv positional encoding
        '''

        super(BridgeNet_FF_v2, self).__init__()
        self.nsample = nsample
        self.downsample = downsample
        self.sample_ch = int(out_ch_list[-1]/f_sample_r)

        # Geometric feature Convolutional layer
        self.pos_mode = extractor_pos_mode
        assert self.pos_mode in ['legacy', 'rot_inv']

        if self.pos_mode == 'legacy':
            self.pos_encoder = geo_feature_encoding
        else:
            self.pos_encoder = GeometricNeigh_Embedding(self.sample_ch, 
                                                        pos_config.coef_d, pos_config.coef_a, 
                                                        normal=pos_config.normal_flag)
        if self.pos_mode == 'rot_inv':
            geo_ch = self.sample_ch

        self.pos_proj = nn.Sequential(nn.Conv2d(geo_ch, self.sample_ch, 1),
                                      nn.BatchNorm2d(self.sample_ch),
                                      activation_func)
        
        self.lme_list = nn.ModuleList()
        for in_ch, out_ch in zip(in_ch_list, out_ch_list):
            lme = LocFeacoding(nsample, in_ch, out_ch,
                               self.sample_ch, activation_func,
                               downsample=downsample)
            self.lme_list.append(lme)

    def forward(self, points, features, grouped_idx):
        """
        Input:
            points: input points position data, [B, N, 3+?]
            features: input points data, [B, N, ch_in]
            grouped_id: supported grouped_id from pre-computation [B, N, 64 or 128]
        Return:
            points: sample points feature data, [B, N, ch_out]
        """
        grouped_points = index_points(points, grouped_idx)
        if self.pos_mode == 'legacy':
            xyz = points[..., 0:3]
            grouped_xyz = grouped_points[..., 0:3]
            expanded_f = self.pos_encode_legacy(xyz, grouped_xyz)
        else:
            expanded_f = self.pos_encode_rotinv(points, grouped_points)

        features = torch.unsqueeze(features.permute(0, 2, 1), dim=-1)

        for i in range(len(self.lme_list)):
            features = self.lme_list[i](expanded_f, features, grouped_idx)           # [B, in_ch, N, 1]

        features = torch.squeeze(features, dim=-1).permute(0, 2, 1)
        return features
    
    def pos_encode_legacy(self, xyz, grouped_xyz):
        expanded_f = self.pos_encoder(xyz, grouped_xyz, self.nsample)
        expanded_f = expanded_f.permute(0, 3, 1, 2)
        expanded_f = self.pos_proj(expanded_f)
        return expanded_f

    def pos_encode_rotinv(self, points, grouped_points):
        expanded_f = self.pos_encoder(points, grouped_points, None)
        expanded_f = expanded_f.permute(0, 3, 1, 2)
        expanded_f = self.pos_proj(expanded_f)
        return expanded_f


class LocFeacoding(nn.Module):
    def __init__(self, nsample, in_ch, out_ch, sample_ch, activation_func, downsample='maxpooling'):
        '''
            Encode the geometric feature
        '''
        super(LocFeacoding, self).__init__()
        group_ch = sample_ch*2
        self.downsample = downsample

        # Sampling layers
        self.sub_f_layer = nn.Sequential(
            nn.Conv2d(in_ch, sample_ch, 1), 
            nn.BatchNorm2d(sample_ch),
            activation_func
        )
        self.up_f_layer = nn.Sequential(
            nn.Conv2d(sample_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch), 
            activation_func
        )
        # GCM
        self.gcm_layer = nn.Sequential(
            nn.Conv2d(group_ch, sample_ch, 1),
            nn.BatchNorm2d(sample_ch),
            activation_func
        )
        # Channel-wise attention
        self.att_conv_catfea = nn.Linear(nsample, 1)      # generate a [nsmaple, 1] matrix to make the concatenated feature attentive
        
        # Residual layer
        self.res_layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            activation_func
        )

    def forward(self, expanded_f, feature, grouped_idx):
        """
        Input:
            expanded_f: input points position data, [B, M, ch_in]
            feature: input points data, [B, ch_in, N, 1]
            grouped_id: supported grouped_id from pre-computation [B, N, k]
        Return:
            points: sample points feature data, [B, N, ch_out]
        """
        # Down-sampling
        sub_feature = self.sub_f_layer(feature)                              # [B, sample_ch, N, 1]
        sub_feature = torch.squeeze(sub_feature, dim=-1).permute(0, 2, 1)    # [B, N, sample_ch]
        expanded_f = expanded_f.permute(0, 2, 3, 1)

        # This is the module which to process the down-sampled tensors
        grouped_sub_f = index_points(sub_feature, grouped_idx)               # [B, N, nsample, sample_ch]
        concat_f = torch.cat([expanded_f, grouped_sub_f], dim=-1)            # [B, N, nsample, sample_ch*2]

        # Geometric Convolutional Modeling (GCM)
        concat_f = concat_f.permute(0, 3, 1, 2)                              # [B, sample_ch*2, N, nsample]
        concat_f = self.gcm_layer(concat_f)
        concat_f = concat_f.permute(0, 2, 3, 1)                              # [B, N, nsample, sample_ch*2]

        # Attentional Aggregation (AG)
        B, N, nsample, ch = concat_f.shape
        # print('concat_f: {}'.format(concat_f.shape))
        concat_fea_score = concat_f.reshape(B*N, nsample, ch).permute(0, 2, 1)
        cat_corr_score = self.att_conv_catfea(concat_fea_score).reshape(B, N, ch)      # [B, N, sample_ch, 1]
        cat_corr_score = torch.softmax(cat_corr_score, dim=-1)                          # [B, N, sample_ch, 1]
        
        if self.downsample == 'sum':
            concat_f = torch.sum(concat_f, dim=2, keepdim=True)              
        elif self.downsample == 'maxpooling':
            concat_f, _ = torch.max(concat_f, dim=2, keepdim=True)           # [B, N, 1, sample_ch]
        elif self.downsample == 'mean':
            concat_f = torch.mean(concat_f, dim=2, keepdim=True)
        
        concat_f_weighted = concat_f*torch.unsqueeze(cat_corr_score, dim=2)  # [B, N, 1, sample_ch]
        concat_f = concat_f + concat_f_weighted                              # residual [B, N, 1, sample_ch]  
        concat_f = concat_f.permute(0, 3, 1, 2)                              # [B, ch_sample, N, 1]
        
        # Up-sampling                
        concat_f = self.up_f_layer(concat_f)                                 # [B, in_ch, N, 1]

        # Output
        res_fs = self.res_layer(feature)                                     # [B, out_ch, N, 1]
        feature = res_fs + concat_f                               
        return feature

class Simple_BridgeNet_FF(nn.Module):
    def __init__(self, in_ch, out_ch, geo_ch, nsample:int, 
                 activation_func, pos_config, 
                 downsample='maxpooling'):
        '''
        This is the BridgeNetv1 feature block which might be used to substitute the 
        feedforward network in original transformer.
        Args:
            radius_list:
            nsample_list:
            in_ch: [], list of int indicates the input channel of the feature
            out_ch: [], list of int indicates the output channel of the feature
            geo_ch: Number of features directly from (x, y, z) coordinates [xi, xj, |xi - xj|, ||xi - xj||]
            f_sample_r: sampling ratio of features
        '''

        super(Simple_BridgeNet_FF, self).__init__()
        self.nsample = nsample
        self.downsample = downsample
        # Geometric feature Convolutional layer
        self.pos_mode = pos_config.pos_mode
        assert self.pos_mode in ['legacy', 'rot_inv']

        if self.pos_mode == 'legacy':
            self.pos_encoder = geo_feature_encoding     # it is en
        else:
            self.pos_encoder = GeometricNeigh_Embedding(in_ch, pos_config.coef_d, 
                                                   pos_config.coef_a)
        if self.pos_mode == 'rot_inv':
            geo_ch = in_ch

        self.pos_proj = nn.Sequential(nn.Conv2d(geo_ch, in_ch, 1),
                                      nn.BatchNorm2d(in_ch),
                                      activation_func)

        self.downsample = downsample
        # GCM
        self.gcm_layer = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            activation_func
        )

        # Channel-wise attention
        self.att_conv_catfea = nn.Sequential(
            # nn.Linear(nsample, nsample),
            nn.Linear(nsample, 1))  
        
        # Output layer
        self.out_layer = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.LayerNorm(out_ch),
            activation_func
        )
    def forward(self, points, features, grouped_idx):
        """
        Input:
            points: input points position data, [B, N, 3+?]
            features: input points data, [B, N, ch_in]
            grouped_id: supported grouped_id from pre-computation [B, N, 64 or 128]
        Return:
            points: sample points feature data, [B, N, ch_out]
        """
        grouped_points = index_points(points, grouped_idx)
        if self.pos_mode == 'legacy':
            expanded_f = self.pos_encode_legacy(points, grouped_points)
        else:
            expanded_f = self.pos_encode_rotinv(points, grouped_points)

        grouped_f = index_points(features, grouped_idx)                       # [B, N, nsample, in_ch]
        # concat_f = torch.concat([grouped_f, expanded_f], dim=-1)            # [B, N, nsample, in_ch]
        concat_f = grouped_f + expanded_f
        # Geometric Convolutional Modeling (GCM)
        concat_f = concat_f.permute(0, 3, 1, 2)                              # [B, sample_ch*2, N, nsample]
        concat_f = self.gcm_layer(concat_f)
        concat_f = concat_f.permute(0, 2, 3, 1)                              # [B, N, nsample, sample_ch*2]

        # Attentional Aggregation (AG)
        cat_corr_score = self.att_conv_catfea(concat_f.transpose(2, 3))      # [B, N, sample_ch, 1]
        cat_corr_score = torch.softmax(cat_corr_score, dim=2)                # [B, N, sample_ch, 1]
        concat_f = concat_f*cat_corr_score.transpose(2, 3)                   # [B, N, nsample, sample_ch]

        if self.downsample == 'sum':
            concat_f = torch.sum(concat_f, dim=2, keepdim=False)              
        elif self.downsample == 'maxpooling':
            concat_f, _ = torch.max(concat_f, dim=2, keepdim=False)           # [B, N, ch_sample]
        elif self.downsample == 'mean':
            concat_f = torch.mean(concat_f, dim=2, keepdim=False)
        
        concat_f = concat_f + features                                        # residual [B, N, ch_sample]  
        # Output
        features = self.out_layer(concat_f)                                   # [B, N, out_ch]
        return features
    
    def pos_encode_legacy(self, points, grouped_points):
        expanded_f = self.pos_encoder(points, grouped_points, self.nsample)
        expanded_f = expanded_f.permute(0, 3, 1, 2)
        expanded_f = self.pos_proj(expanded_f)
        expanded_f = expanded_f.permute(0, 2, 3, 1) 
        return expanded_f

    def pos_encode_rotinv(self, points, grouped_points):
        expanded_f = self.pos_encoder(points, grouped_points, None)
        return expanded_f


class Naive_BridgeNet_FF(nn.Module):
    def __init__(self, in_ch, out_ch, geo_ch, nsample:int, 
                 activation_func, pos_config, 
                 downsample='maxpooling'):
        '''
        This is the BridgeNetv1 feature block which might be used to substitute the 
        feedforward network in original transformer.
        Args:
            radius_list:
            nsample_list:
            in_ch: [], list of int indicates the input channel of the feature
            out_ch: [], list of int indicates the output channel of the feature
            geo_ch: Number of features directly from (x, y, z) coordinates [xi, xj, |xi - xj|, ||xi - xj||]
            f_sample_r: sampling ratio of features
        '''

        super(Naive_BridgeNet_FF, self).__init__()
        self.nsample = nsample
        self.downsample = downsample
        # Geometric feature Convolutional layer
        self.pos_mode = pos_config.pos_mode
        assert self.pos_mode in ['legacy', 'rot_inv']

        self.pos_encoder = geo_feature_encoding
        self.pos_proj = nn.Sequential(nn.Linear(geo_ch, in_ch),)
        self.downsample = downsample

        self.gcm_layer = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            activation_func
            )
        # Channel-wise attention
        self.att_conv_catfea = nn.Sequential(
            # nn.Linear(nsample, nsample),
            nn.Linear(nsample, 1))  
        
        # Output layer
        self.out_layer = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.LayerNorm(out_ch),
            activation_func
        )
    def forward(self, points, features, grouped_idx):
        """
        Input:
            points: input points position data, [B, N, 3+?]
            features: input points data, [B, N, ch_in]
            grouped_id: supported grouped_id from pre-computation [B, N, 64 or 128]
        Return:
            points: sample points feature data, [B, N, ch_out]
        """
        grouped_points = index_points(points, grouped_idx)
        xyz = points[..., 0:3]
        grouped_xyz = grouped_points[..., 0:3]
        
        expanded_f = self.pos_encode_legacy(xyz, grouped_xyz)

        grouped_f = index_points(features, grouped_idx)                       # [B, N, nsample, in_ch]
        concat_f = grouped_f + expanded_f
        concat_f = self.gcm_layer(concat_f)
        if self.downsample == 'sum':
            concat_f = torch.sum(concat_f, dim=2, keepdim=False)              
        elif self.downsample == 'maxpooling':
            concat_f, _ = torch.max(concat_f, dim=2, keepdim=False)           # [B, N, ch_sample]
        elif self.downsample == 'mean':
            concat_f = torch.mean(concat_f, dim=2, keepdim=False)
        
        concat_f = concat_f + features                                        # residual [B, N, ch_sample]  
        # Output
        features = self.out_layer(concat_f)                                   # [B, N, out_ch]
        return features
    
    def pos_encode_legacy(self, points, grouped_points):
        expanded_f = self.pos_encoder(points, grouped_points, self.nsample)
        # expanded_f = expanded_f.permute(0, 3, 1, 2)
        expanded_f = self.pos_proj(expanded_f)
        # expanded_f = expanded_f.permute(0, 2, 3, 1) 
        return expanded_f

    def pos_encode_rotinv(self, points, grouped_points):
        expanded_f = self.pos_encoder(points, grouped_points, None)
        return expanded_f

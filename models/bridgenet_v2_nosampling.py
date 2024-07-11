import torch.nn.functional as F

from torch import nn
from models.modules.transformer import NeighborGeometricAttention, Global_Att
from models.modules.feature_extractor import BridgeNet_FF_v2
from models.modules.upsampling import BridgeNetUp_knn
from models.modules.utils import get_activation_func, points_sampling, \
    knn_algorithm, ball_query_algorithm

class Bridge_Netv2(nn.Module):
    def __init__(self, in_ch_lists, out_ch_lists, up_in_ch_list, up_mlp_ch_lists, 
                 extractor_mode, 
                 agg_in_ch, agg_out_ch, 
                 neigh_mode, agg_nsample_list, extractor_nsample_list, radius_list, 
                 out_dropout: float, activation: str, n_cls: int, geo_ch, 
                 sampling_method: str, sampling_ratio: list[int], 
                 f_sample_r_list,
                 pre_norm: bool,
                 pos_config,
                 block_config,
                 model_n):
        super().__init__()
        assert len(in_ch_lists) == len(out_ch_lists)
        activation_func = get_activation_func(activation)

        self.agg_nsample_list = agg_nsample_list
        self.extractor_nsample_list = extractor_nsample_list
        self.radius_list = radius_list
        self.neigh_mode = neigh_mode
        self.model_n = model_n

        assert self.model_n in ['bridgenetv1', 'bridgenetv2']
        assert self.neigh_mode in ['knn', 'ball_query']
        assert extractor_mode in ['simple', 'bridgenet_v1']

        self.sampling_ratio = sampling_ratio
        self.sampling_method = sampling_method
        self.n_sub_blocks = [len(in_ch_list) for in_ch_list in in_ch_lists]

        # Define the downsampling layers
        self.embedding_in = nn.Sequential(
            nn.Conv1d(3, in_ch_lists[0][0], 1),
            nn.BatchNorm1d(in_ch_lists[0][0]), 
            activation_func,)

        # Define the subsampling layers
        self.agg_feature = nn.ModuleList()
        self.ff_blocks_list = nn.ModuleList()

        for i, (in_ch_list, out_ch_list) in enumerate(zip(in_ch_lists, out_ch_lists)):
            agg_nsample = agg_nsample_list[i]
            extractor_nsample = self.extractor_nsample_list[i]
            if extractor_mode == 'bridgenet_v1':
                f_sample_r = f_sample_r_list[i]
            self.agg_feature.append(NeighborGeometricAttention(att_config=block_config.dense_attention,
                                    in_ch=agg_in_ch[i], out_ch=agg_out_ch[i], n_head=block_config.n_head, 
                                    geo_ch=geo_ch, nsample=agg_nsample,
                                    att_dropout=block_config.att_dropout, activation_func=activation_func,
                                    pos_config=pos_config))
            ff_blocks = nn.ModuleList()
            ff_blocks = BridgeNet_FF_v2(in_ch_list=in_ch_list, out_ch_list=out_ch_list, geo_ch=geo_ch,
                                        f_sample_r=f_sample_r, nsample=extractor_nsample, 
                                        activation_func=activation_func, 
                                        downsample=block_config.downsample,
                                        pos_config=pos_config, extractor_pos_mode='legacy')
 
            self.ff_blocks_list.append(ff_blocks)
        
        # Global feature extractor
        self.glob_att = Global_Att(out_ch_list[-1], 8, pre_norm, activation_func, 
                                   att_config=block_config.dense_attention)

        # Define the upsampling layers
        self.upsampling_blocks = nn.ModuleList()
        for up_in_ch, up_mlp_ch in zip(up_in_ch_list, up_mlp_ch_lists):
            self.upsampling_blocks.append(
                BridgeNetUp_knn(in_ch=up_in_ch, mlp=up_mlp_ch))
        
        self.embedding_out = nn.Sequential(
            nn.Conv1d(up_mlp_ch_lists[-1][-1], up_mlp_ch_lists[-1][-1]*2, 1),
            nn.BatchNorm1d(up_mlp_ch_lists[-1][-1]*2), 
            activation_func,
            nn.Conv1d(up_mlp_ch_lists[-1][-1]*2, up_mlp_ch_lists[-1][-1], 1),
            nn.BatchNorm1d(up_mlp_ch_lists[-1][-1]), 
            activation_func,
            nn.Dropout(out_dropout), 
            nn.Conv1d(up_mlp_ch_lists[-1][-1], n_cls, 1),
            nn.BatchNorm1d(n_cls),
            )

    def forward(self, points):
        xyzs = points[..., 0:3]
        xyzs = xyzs.transpose(1, 2)
        feature = self.embedding_in(xyzs)
        feature = feature.transpose(1, 2)
        feature = self.bridgenetv2(points, feature)
        return feature

    def bridgenetv2(self, points, feature):
        B, N, _ = points.shape
        n_points_sampled = [int(N/i) for i in self.sampling_ratio]

        feature_encode_list = []
        points_list = []
        points_list.append(points) 

        for i in range(len(self.ff_blocks_list)):
            points1 = points_list[-1]
            xyzs1 = points1[..., 0:3]
            extractor_nsample = self.extractor_nsample_list[i]
            radius = self.radius_list[i]
            neigh_idx_self = ball_query_algorithm(radius, extractor_nsample, xyzs1, xyzs1) # bridgenet block (ball query)
            neigh_idx_self = neigh_idx_self.detach()
            # (1) Feature Extractor
            if i == 0:
                feature1 = feature           # take the input as the feature for the first layer
            else:
                feature1 = feature_encode_list[-1]
            ff_block = self.ff_blocks_list[i]
            feature1 = ff_block(points_list[-1], feature1, neigh_idx_self)
            if i == 0:
                feature_encode_list.append(feature1)
            # sample point based on the index for the original point cloud
            # sample feature based on the index for the already sampled features
            _, points2, feature2 = points_sampling(self.sampling_method, n_points_sampled[i], 
                                                   points1, feature1)
            points_list.append(points2)

            # (2) Local Geo-Transformer
            xyzs2 = points2[..., 0:3]
            agg_nsample = self.agg_nsample_list[i]
            neigh_idx_agg = knn_algorithm(agg_nsample, xyzs1, xyzs2) # local transformer (knn)
            neigh_idx_agg = neigh_idx_agg.detach()
            feature = self.agg_feature[i](points1, points2, feature1, feature2, neigh_idx_agg)
            if i < (len(self.ff_blocks_list)-1):  
                feature_encode_list.append(feature)

        feature = self.glob_att(feature, points_list[-1])
        # Up-sampling
        up_idx = [i for i in range(-1, -(len(points_list)+1), -1)]
        for i in range(len(self.upsampling_blocks)):
            upsampling_block = self.upsampling_blocks[i]
            feature_up = feature_encode_list[up_idx[i]]
            feature = upsampling_block(feature, feature_up, 
                                       points_list[up_idx[i]], points_list[up_idx[i+1]])
        feature = feature.transpose(1, 2)
        feature = self.embedding_out(feature)
        feature = feature.transpose(1, 2)
        feature = F.log_softmax(feature, dim=-1)
        return feature


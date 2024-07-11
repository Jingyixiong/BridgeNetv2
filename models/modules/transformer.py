import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.modules.mem_eff_similarity.geoformer_fdimatt import Geoneigh_att, Geoneigh_GeoAtt, \
     Geoneigh_FPT, Geoneigh_FPT_v2, Geoglobal_FPT
from models.modules.utils import index_points, pairwise_distance

from models.modules.utils import index_points
from models.modules.postional_encode import Pos_Encoding, Pos_EncodingCoords, GeometricNeigh_Embedding 

class Cross_Attention(nn.Module):
    '''
    This function defines the self transformer layer, no mask is considered.
    Parameters:
        in_channel: input channel
        num_head: number of attention head
        dropout: dropout layer for the output, to avoid overfitting
        att_dropout: drop out ratio for attention weight
    Input:
        query: [B, n_query, dim]
        key: [B, N, dim]
        value: [B, N, dim] 
        pos_encode_q: [B, n_query, dim], positional encoding for query
        pos_encode_k: [B, N, dim], positional encoding for key
    Output:
        query: [B, N, dim]
    '''
    def __init__(self, in_ch:int, num_head:int, activation_func, dropout:float=0.0, att_dropout:float=0.0,):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=in_ch, 
                                                    num_heads=num_head,
                                                    dropout=att_dropout,
                                                    batch_first=True)
        self.layer_norm = nn.LayerNorm(in_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, 
                pos_encode_q: Tensor, pos_encode_k: Tensor):
        # consider positional encoding, the way how it is combined with qkv requires more trials
        query = query + pos_encode_q     
        key = key + pos_encode_k
        query_attn, _ = self.multihead_attn(query=query, key=key, value=value)
        query = query + self.dropout(query_attn)
        query = self.layer_norm(query)
        return query 

class Global_Att(nn.Module):
    '''
    This function implements self attention with geometric 
    multi-attention mechanism
    Parameters:
        in_channel: input channel
        num_head: number of attention head
        dropout: dropout layer for the output, to avoid overfitting
        att_dropout: drop out ratio for attention weight
        pos_mode: the way of combining postional encoding. 'deactivate', 'add', 'concat'
    Input:
        qkv: [B, N, dim], input is treated as query, key and value 
        pos_encode: [B, N, dim], positional encoding(learnable or not)
    Output:
        qkv: [B, N, dim]
    '''
    def __init__(self, in_ch:int, n_head, norm_before, activation_func, att_config,
                 dropout:float=0.0, att_dropout:float=0.0):
        super().__init__()
        
        self.att_dropout = att_dropout
             
        self.n_head = n_head
        if self.n_head:
            self.hdim = in_ch//self.n_head if in_ch//self.n_head > 1 else 1
        
        self.linear_k = nn.Linear(in_ch, self.hdim*self.n_head)
        self.linear_v = nn.Linear(in_ch, self.hdim*self.n_head)
        self.linear_q = nn.Linear(in_ch, self.hdim*self.n_head)

        self.multi_head_att = Geoglobal_FPT(self.n_head, self.hdim, att_config)

        self.linear_out = nn.Linear(self.hdim*self.n_head, in_ch)
        self.layer_norm = nn.LayerNorm(in_ch)
        self.dropout = nn.Dropout(dropout)

        self.ffn = FeedForward(in_ch=in_ch, hidden_ch=in_ch*2, out_ch=in_ch, 
                               dropout=dropout, norm_before=norm_before, 
                               activation_func=activation_func)

    def forward(self, feature: Tensor, points: Tensor):
        # consider positional encoding, the way how it is combined with qkv requires more trials
        B, N, _ = feature.shape
        in_feature = feature
        xyz = points[..., 0:3]

        expanded_f =  pairwise_distance(x=xyz, y=xyz)      # [B, N, N]
        q = self.linear_q(in_feature)                      # [B, N, self.hdim*self.n_head]
        k = self.linear_k(in_feature)                      # [B, N, self.hdim*self.n_head]
        v = self.linear_v(in_feature)                      # [B, N, self.hdim*self.n_head]
        if self.n_head:
            q = q.view(B, N, self.n_head, self.hdim)
            k = k.view(B, N, self.n_head, self.hdim)
            v = v.view(B, N, self.n_head, self.hdim)

        in_feature = self.multi_head_att(q, k, v, expanded_f)
        in_feature = in_feature.view(B, N, -1)
        feature = feature + self.dropout(in_feature)
        feature = self.layer_norm(feature)
        return feature

class NeighborGeometricAttention(nn.Module):
    def __init__(self, att_config ,in_ch, out_ch, n_head, geo_ch, nsample:int,
                 att_dropout, activation_func, pos_config):
        '''
        Args:
            radius_list:
            nsample_list:
            in_ch: [], list of int indicates the input channel of the feature
            out_ch: [], list of int indicates the output channel of the feature
            n_feature: Number of features directly from (x, y, z) coordinates [xi, xj, |xi - xj|, ||xi - xj||]
            geo_fea_up: if True, it first make the last dim of the geo-features the same as the input feature tensor
        '''
        super(NeighborGeometricAttention, self).__init__()
        self.nsample = nsample
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_head = n_head

        self.head_ch = in_ch//self.n_head if in_ch//self.n_head > 1 else in_ch
        
        # Define the positional encoding
        self.pos_mode = pos_config.pos_mode
        if self.pos_mode == 'coords':
            self.pos_encoder = nn.Sequential(
                nn.Linear(3, in_ch),
                activation_func
            )
        elif self.pos_mode == 'rela':
            self.pos_encoder = nn.Sequential(
                nn.Linear(1, in_ch),
                activation_func
            )
        elif self.pos_mode == 'rot_inv': 
            self.pos_encoder = GeometricNeigh_Embedding(in_ch, pos_config.coef_d, pos_config.coef_a, 
                                                        normal=pos_config.normal_flag)
            
        self.linear_k = nn.Linear(self.in_ch, self.in_ch)
        self.linear_v = nn.Linear(self.in_ch, self.in_ch)
        self.linear_q = nn.Linear(self.in_ch, self.in_ch) 

        self.multi_head_att = Geoneigh_FPT(self.n_head, self.head_ch, att_config)
    
        self.att_norm = nn.LayerNorm(in_ch)    # same dim for both post and pre norm
        self.att_dropout = nn.Dropout(att_dropout)
        
        self.layer_out = nn.Sequential(
            nn.Linear(self.in_ch, self.out_ch),
            nn.LayerNorm(self.out_ch)
            )

    def forward(self, points1, points2, features1, features2, grouped_idx):
        """
        Input:
            points1: input points position data, [B, N, 3+?]
            points2: input points position data, [B, M, 3+?], note that N > M
            features1: input points data, [B, N, ch_in]
            features2: input points data, [B, M, ch_in]
            grouped_id: supported grouped_id from pre-computation [B, N, 64 or 128]
        Return:
            points: sample points feature data, [B, N, ch_out]
        """
        device = features2.device
        B, M, _ = features2.shape

        grouped_points = index_points(points1, grouped_idx)             # [B, M, nsample, 3 + 3]
        if self.pos_mode == 'rot_inv':
            expanded_f = self.pos_encoder(points2, grouped_points,)
        elif self.pos_mode == 'null':
            expanded_f = torch.zeros(1).to(device)
        else:
            expanded_f = self.legacy_pos_encoding(points2, grouped_points, self.pos_mode)
            # print('Before: {}'.format(expanded_f.shape))
            expanded_f = self.pos_encoder(expanded_f)
            # print('After: {}'.format(expanded_f.shape))
        # Cross Attention module
        q = self.linear_q(features2)
        k = self.linear_k(features1)
        v = self.linear_v(features1)
        k_neigh = index_points(k, grouped_idx)
        v_neigh = index_points(v, grouped_idx)
        v_neigh = v_neigh + expanded_f
        agg_feature = self.multi_head_att(q, k_neigh, v_neigh, expanded_f)
        agg_feature = agg_feature.view(B, M, -1)
        feature = self.att_norm(features2 + self.att_dropout(agg_feature))
        
        feature = self.layer_out(feature)

        return feature

    def legacy_pos_encoding(self, points, grouped_points, n_pos):
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
        rela_xyz = xyz_expanded - grouped_xyz   
        if n_pos == 'coords':
            return rela_xyz
        elif n_pos == 'rela': 
            rela_dis = torch.linalg.norm(rela_xyz, dim = -1, keepdim=True)            # [B, N, nsample, 1]
            return rela_dis
        else:
            raise Exception('The given n_pos does not fit in the domian!')

class FeedForward(nn.Module):
    '''
    This function defines the feedforward layers after Multihead attentions.
    Input: 
        feature: [B, N, in_ch]
    Output:
        feature: [B, N, in_ch]
    '''
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, 
                 dropout: float, norm_before=False, activation_func=nn.ReLU):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        if self.in_ch != self.out_ch:
            self.linear_skip = nn.Linear(in_ch, out_ch)
        self.norm_before = norm_before
        if self.norm_before:
            self.norm = nn.LayerNorm(in_ch)
        else:
            self.norm = nn.LayerNorm(out_ch)
    
        self.linear_1 = nn.Linear(in_ch, hidden_ch)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_ch, out_ch)
        self.activation_func = activation_func

    def forward(self, feature):
        if self.norm_before:
            feature = self.pre_ln(feature)
        else:
            feature = self.post_ln(feature)
        return feature

    def post_ln(self, feature):
        skip_f  = feature
        feature = self.activation_func(self.linear_1(feature))
        feature = self.linear_2(feature)
        feature = self.dropout(feature) + skip_f
        feature = self.norm(feature)
        return feature
    
    def pre_ln(self, feature):
        norm_f = self.norm(feature)
        norm_f = self.activation_func(self.linear_1(norm_f))
        norm_f = self.linear_2(norm_f)
        feature = self.dropout(norm_f) + feature
        return feature
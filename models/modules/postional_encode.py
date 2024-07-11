import numpy as np
import torch
from torch import nn, Tensor

from models.modules.utils import pairwise_distance, index_points

class Pos_Encoding(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, learnable:bool = True):
        """
        This function implement the positional encoding method defined in 
        'Attention is all you need'
        Parameters:
            out_ch: the output channel dim
        Input:
            coords: [B, N, 3], 3 means x, y, z
        Output:
            pos_encode: [B, N, out_ch]
        """
        super().__init__()
        self.in_ch = in_ch
        self.coord_ch = int(np.ceil(out_ch/3))   # the dimension for every single coordinate channel
        if self.coord_ch%2:
            self.coord_ch += 1
        self.out_ch = out_ch
        invt_div_term = 1.0 / (10000**(torch.arange(0, self.coord_ch, 2).float()/self.coord_ch))
        self.register_buffer("invt_div_term", invt_div_term)
        
        self.learnable = learnable
        if self.learnable:
            self.pos_learn = nn.Linear(self.coord_ch*3, out_ch)

    def forward(self, coords: Tensor) -> Tensor:
        B, N, ch = coords.shape
        device = coords.device
        assert self.in_ch == ch 
        pe_xyz = []
        for i in range(ch):
            coord = coords[:, :, i]
            pe  = torch.zeros(B, N, self.coord_ch)
            pe[:, :, 0::2] = torch.sin(torch.einsum('bn,c->bnc', coord, self.invt_div_term))
            pe[:, :, 1::2] = torch.cos(torch.einsum('bn,c->bnc', coord, self.invt_div_term))
            pe_xyz.append(pe)
        pe_xyz = torch.concatenate(pe_xyz, dim=-1).to(device)
        if self.learnable:
            pe_xyz = self.pos_learn(pe_xyz)
        return pe_xyz[:, :, :self.out_ch]


class Pos_EncodingCoords(nn.Module):
    '''
    This function simply uses naive coordiates as positional encoding.
    '''
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.pos_learn = nn.Linear(self.in_ch, out_ch)
    
    def forward(self, coords: Tensor) -> Tensor:
        assert coords.shape[-1] == self.in_ch

        pe_xyz = self.pos_learn(coords)
        return pe_xyz


class SinusoidalPositionalEmbedding(nn.Module):
    '''
    This module is adopted from 
    'https://github.com/qinzheng93/GeoTransformer/blob/main/geotransformer/modules/transformer/positional_embedding.py'
    '''
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        """Sinusoidal Positional Embedding.
        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings

class GeometricNeigh_Embedding(nn.Module):
    def __init__(self, in_ch, coef_d, coef_a, normal=False, reduction='maxpooling'):
        super().__init__()
        self.in_ch = in_ch
        self.coef_d = coef_d
        self.coef_a = coef_a
        self.factor_a = 180.0 / (self.coef_a * np.pi)   # readjust the weight of angle
        self.reduction = reduction
        self.normal = normal

        self.embedding = SinusoidalPositionalEmbedding(in_ch)
        self.proj_d = nn.Linear(in_ch, in_ch)
        self.proj_a = nn.Linear(in_ch, in_ch)

    def forward(self, points, grouped_points, nsample=None):
        '''
        Input:
            points: [B, N, 3+?]
            grouped_points: [B, N, 3+?]
        '''
        '''
        Input:
            points: [B, N, 3 + ?]  --> there could be probably other geometrical features to be added
            grouped_points: [B, N, nsample, 3 + ?]
        Output:
            pos_embedding: [B, N, nsample, n_fs]
        '''
        d_indices, a_indices = self.local_geo_embedding(points, grouped_points)
        d_embeddings = self.embedding(d_indices)     # [B, N, k, h_dim]
        d_embeddings = self.proj_d(d_embeddings)
        if self.normal:
            a_embeddings = self.embedding(a_indices)   # [B, N, k, h_dim]
            a_embeddings = self.proj_a(a_embeddings)   
            # if self.reduction == 'sum': 
            #     a_embeddings = torch.sum(a_embeddings, dim=2)                      
            # elif self.reduction == 'maxpooling':
            #     a_embeddings, _ = torch.max(a_embeddings, dim=2)  
            #     print('The a_embeddings has the shape of: {}'.format(a_embeddings.shape))            
            # elif self.reduction == 'mean':
            #     a_embeddings = torch.mean(a_embeddings, dim=2)
            pos_embedding = d_embeddings + a_embeddings
        else:
            # print('Only relative distance is used as extra information!')
            pos_embedding = d_embeddings
        return pos_embedding

    @torch.no_grad()
    def local_geo_embedding(self, points, grouped_points):

        xyz = points[..., 0:3]
        grouped_xyz = grouped_points[..., 0:3]
        
        B, N, _ = xyz.shape
        _, _, nsample, _ = grouped_xyz.shape
        xyz_expanded = torch.unsqueeze(xyz, dim=2).expand(B, N, nsample, 3)      # [B, N, nsample, 3]
        rela_xyz = xyz_expanded - grouped_xyz                                     # [B, N, nsample, 3]
        rela_dis = torch.linalg.norm(rela_xyz, dim = -1, keepdim=True)            # [B, N, nsample, 1]
        # norm_rela_xyz = rela_xyz/rela_dis                                       # [B, N, nsample, 3]

        # calculate the angle between pairs of neighbor points and centre point
        if self.normal:
            normals = points[..., 3:]
            grouped_normals = grouped_points[..., 3:]
            cos_values = torch.sum(normals.unsqueeze(dim=2)*grouped_normals, dim=-1)  # [B, N, nsample]
            cross_x = normals.unsqueeze(dim=2).expand(-1,- 1, nsample, -1)
            cross_y = grouped_normals
            sin_values = torch.norm(torch.cross(cross_x, cross_y, dim=-1), dim=-1)    # [B, N, nsample]    

            angles = torch.atan2(sin_values, cos_values)
            a_value = angles*self.factor_a 
            
            return rela_dis.squeeze(dim=-1)/self.coef_d, a_value

        else:
            return rela_dis.squeeze(dim=-1)/self.coef_d, 0

class GeometricGlob_Embedding_Original(nn.Module):
    def __init__(self, in_ch, coef_d, coef_a, angle_k, normal=False, downsample='maxpooling'):
        super(GeometricGlob_Embedding_Original, self).__init__()
        self.coef_d = coef_d
        self.coef_a = coef_a
        self.factor_a = 180.0 / (self.coef_a * np.pi)
        self.angle_k = angle_k
        self.normal = normal

        self.embedding = SinusoidalPositionalEmbedding(in_ch)
        self.proj_d = nn.Linear(in_ch, in_ch)
        self.proj_a = nn.Linear(in_ch, in_ch)

        self.downsample = downsample
        if self.downsample not in ['maxpooling', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.downsample}.')

    @torch.no_grad()
    def get_embedding_indices(self, xyz):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            xyz: torch.Tensor (B, N, 3)

        Returns:
            d_indices: torch.FloatTensor (B, N, k_neigh), distance embedding indices
            a_indices: torch.FloatTensor (B, N, k_neigh, angle_k), angular embedding indices
        """
        B, N, _ = xyz.shape

        dist_map = torch.sqrt(pairwise_distance(xyz, xyz))                                 # (B, N, N)
        d_indices = dist_map / self.coef_d

        k = self.angle_k
        neigh_idx = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]              # (B, N, k)
        grouped_points = index_points(xyz, neigh_idx)

        ref_vectors = grouped_points - xyz.unsqueeze(2)                                    # (B, N, k, 3)
        norm_ref_vectors = ref_vectors/torch.norm(ref_vectors, dim=-1, keepdim=True)
        anc_vectors = xyz.unsqueeze(1) - xyz.unsqueeze(2)                                  # (B, N, N, 3)
        norm_anc_vectors = anc_vectors/torch.norm(anc_vectors, dim=-1, keepdim=True)

        norm_ref_vectors = norm_ref_vectors.unsqueeze(2).expand(B, N, N, k, 3)             # (B, N, N, k, 3)
        norm_anc_vectors = norm_anc_vectors.unsqueeze(3).expand(B, N, N, k, 3)             # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(norm_ref_vectors, norm_anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(norm_ref_vectors * norm_anc_vectors, dim=-1)                # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)                                       # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, xyz):
        '''
        Input: 
            xyz: [B, N, 3]
        Return:
            embeddings: [B, N, N, in_ch]
        '''
        d_indices, a_indices = self.get_embedding_indices(xyz)
        print(d_indices.shape, a_indices.shape)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        if self.normal:
            a_embeddings = self.embedding(a_indices)
            a_embeddings = self.proj_a(a_embeddings)
            if self.downsample == 'maxpooling':
                a_embeddings = a_embeddings.max(dim=3)[0]
            else:
                a_embeddings = a_embeddings.mean(dim=3)

            embeddings = d_embeddings + a_embeddings
        else:
            embeddings = d_embeddings

        return embeddings


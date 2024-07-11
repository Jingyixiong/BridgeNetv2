import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.modules.utils import pairwise_distance

class Geoneigh_att(nn.Module):
    def __init__(self, nsmaple, config):
        super().__init__()
        self.q_bucket_size = config.q_bucket_size
        self.eps = config.eps
        self.dropout = config.dropout
        self.weight_kernal = config.weight_kernal
        self.att_conv_fea = nn.Sequential(nn.Linear(nsmaple, nsmaple),
                                          nn.Linear(nsmaple, 1))    
            
    def forward(self,q: Tensor, k_neigh: Tensor, v_neigh: Tensor, 
                geo_f: Tensor):
        '''
        Mememory efficient attention block.
        Input:
            config: Dict which contains parameters
            points: [B, n, 3]
            q: [B, n, n_head, hdim]
            k: [B, n, n_head, hdim]  
            v: [B, n, n_head, hdim]  
        '''
        B, N, n_head, hdim = q.shape
        nsample =  k_neigh.shape[2]

        if q.shape[1] > self.q_bucket_size:
            self.q_bucket_size = q.shape[1]
        
        scale = q.shape[-1]**-0.5
        q = q * scale
        k_neigh = k_neigh.view(B, N, nsample, n_head, hdim)                       # [B, N, npoint, n_head, h_dim]
        v_neigh = v_neigh.view(B, N, nsample, n_head, hdim)                       # [B, N, npoint, n_head, h_dim]
        q_chunks = q.split(self.q_bucket_size, dim = 1)
        k_chunks = k_neigh.split(self.q_bucket_size, dim = 1)
        v_chunks = v_neigh.split(self.q_bucket_size, dim = 1)
        if torch.is_tensor(geo_f):
            use_geo_flag = True
            geo_f_chunks = geo_f.split(self.q_bucket_size, dim=1)
        
        # loop through all chunks
        out = []
        for i, (q_chunk, k_chunk, v_chunk) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
            if self.weight_kernal == 'substract':
                weight = q_chunk.unsqueeze(dim=2) - k_chunk                         # [B, n, nsample, n_head, hdim]
                if use_geo_flag:
                    geo_f_chunk = geo_f_chunks[i]
                    geo_f_chunk = geo_f_chunk.unsqueeze(dim=3).expand(-1, -1, -1, n_head, -1)
                    weight_expand = torch.concat([weight, geo_f_chunk], dim=-1)     # [B, n, nsample, n_head, hdim+4]
                    att_score = self.att_conv_fea(weight_expand.transpose(-1, 2))   # [B, n, hdim+4, n_head, 1]
                    weight_expand = weight_expand * att_score.transpose(-1, 2)      # [B, n, nsample, n_head, hdim+4]
                weight = torch.sum(weight_expand, dim=-1)          # [B, n, nsample, n_head]
            else:
                weight = torch.sum(q_chunk.unsqueeze(dim=2)*k_chunk, dim=-1)
            # numerical stability
            weight_max = weight.amax(dim = 2, keepdim = True).detach()                # max in nsample dim
            weight = weight - weight_max
            exp_weight = torch.softmax(weight, dim=2)                                 # [B, n, nsample, n_head]
            exp_weight = F.dropout(exp_weight, p = self.dropout)

            weighted_value = torch.sum(exp_weight.unsqueeze(dim=-1)*v_chunk, dim=2)   # [B, n, n_head, hdim]
            out.append(weighted_value)

        out = torch.cat(out, dim = 1)
        return out

class Geoneigh_GeoAtt(nn.Module):
    def __init__(self, nsmaple, config):
        super().__init__()
        self.q_bucket_size = config.q_bucket_size
        self.eps = config.eps
        self.dropout = config.dropout
        self.weight_kernal = config.weight_kernal
        self.att_conv_fea = nn.Sequential(nn.Linear(nsmaple, nsmaple),
                                          nn.Linear(nsmaple, 1))    
            
    def forward(self,q: Tensor, k_neigh: Tensor, v_neigh: Tensor, 
                geo_f: Tensor):
        '''
        Mememory efficient attention block.
        Input:
            config: Dict which contains parameters
            points: [B, n, 3]
            q: [B, n, n_head, hdim]
            k: [B, n, k, n_head, hdim]  
            v: [B, n, k, n_head, hdim]  
            geo_f: [B, n, k, n_head, hdim]
        '''
        B, N, n_head, hdim = q.shape
        nsample =  k_neigh.shape[2]

        if q.shape[1] > self.q_bucket_size:
            self.q_bucket_size = q.shape[1]
        
        scale = q.shape[-1]**-0.5
        q_chunks = q.split(self.q_bucket_size, dim = 1)
        k_chunks = k_neigh.split(self.q_bucket_size, dim = 1)
        v_chunks = v_neigh.split(self.q_bucket_size, dim = 1)
        geo_f_chunks = geo_f.split(self.q_bucket_size, dim = 1)
        
        # loop through all chunks
        out = []
        for _, (q_chunk, k_chunk, v_chunk, geo_f_chunk) in enumerate(zip(q_chunks, k_chunks, v_chunks ,geo_f_chunks)):
            if self.weight_kernal == 'substract':
                qk_weight = q_chunk.unsqueeze(dim=2) - k_chunk                    # [B, n, nsample, n_head, hdim]
                qgeo_weight = q_chunk.unsqueeze(dim=2) - geo_f_chunk              # [B, n, nsample, n_head, hdim]
                total_weight = (qk_weight + qgeo_weight) / scale  
                # att_score = self.att_conv_fea(weight_expand.transpose(-1, 2))   # [B, n, hdim+4, n_head, 1]
                # weight_expand = weight_expand * att_score.transpose(-1, 2)      # [B, n, nsample, n_head, hdim]
                # weight = torch.sum(total_weight, dim=-1)                          # [B, n, nsample, n_head]
            elif self.weight_kernal == 'cos':
                # element-wise multiplication 
                qk_weight = q_chunk.unsqueeze(dim=2)*k_chunk                      # [B, n, nsample, n_head, hdim]
                qgeo_weight = q_chunk.unsqueeze(dim=2)*geo_f_chunk                # [B, n, nsample, n_head, hdim]
                total_weight = (qk_weight + qgeo_weight) / scale  
            # numerical stability
            weight_max = total_weight.amax(dim = 2, keepdim = True).detach()      # max in nsample dim
            total_weight = total_weight - weight_max
            exp_weight = torch.softmax(total_weight, dim=2)                       # [B, n, nsample, n_head, hdim]
            exp_weight = F.dropout(exp_weight, p = self.dropout)

            weighted_value = torch.sum(exp_weight*v_chunk, dim=2)                  # [B, n, n_head, hdim]
            out.append(weighted_value)

        out = torch.cat(out, dim = 1)
        return out


class Geoglobal_FPT(nn.Module):
    def __init__(self, n_head, hdim, config):
        super().__init__()
        self.n_head = n_head
        self.hdim = hdim
        self.eps = config.eps
        self.dropout = config.dropout
        # self.linear_weight = nn.Sequential(nn.Linear(self.n_head*self.hdim , 
        #                                              self.hdim),
        #                                    nn.ReLU(),
        #                                    nn.LayerNorm(self.hdim))    
            
    def forward(self, q: Tensor, k: Tensor, v: Tensor, 
                geo_f: Tensor):
        '''
        Attention in global scale. Please note that the
        size should not be very large! Better to be 512
        or even smaller.
        Input:
            config: Dict which contains parameters
            points: [B, N, 3]
            q: [B, N, n_head*hdim] if self.n_head != None, else [B, N, n_head, hdim]
            k: [B, N, n_head*hdim]  
            v: [B, N, n_head*hdim]  
            geo_f: [B, n, n, n_head*hdim]
        '''
        B, N, _, _ = k.shape
        scale = q.shape[-1]**-0.5

        if not self.n_head:        # multi-head att is not adopted 
            assert q.ndim == 3
            qk_weight = pairwise_distance(q, k)+geo_f              # [B, N, N]
        else:
            q = q.transpose(1, 2)                                  # [B, n_head, N, hdim]
            k = k.transpose(1, 2)                                  # [B, n_head, N, hdim]
            qk_weight = pairwise_distance(q, k)+geo_f.unsqueeze(1) # [B, n_head, N, N]
        total_weight = (qk_weight)/scale                                              
        # total_weight = self.linear_weight(total_weight)          # [B, n_head, N, N]
        # numerical stability
        weight_max = total_weight.max().detach()                   # global max
        total_weight = total_weight - weight_max
        exp_weight = torch.softmax(total_weight, dim=-1)         
        exp_weight = F.dropout(exp_weight, p = self.dropout)

        if self.n_head:
            v = v.view(B, N, self.n_head, self.hdim)
            weighted_value = torch.einsum('b n h d, b h n m->b n h d', 
                                           v, exp_weight)          # [B, n, n_head, hdim]
        return weighted_value


class Geoneigh_FPT(nn.Module):
    def __init__(self, n_head, hdim, config):
        super().__init__()
        self.n_head = n_head
        self.hdim = hdim
        self.q_bucket_size = config.q_bucket_size
        self.eps = config.eps
        self.dropout = config.dropout
        self.weight_kernal = config.weight_kernal
        self.linear_weight = nn.Sequential(nn.Linear(int(n_head*hdim), hdim),
                                           nn.ReLU(),
                                           nn.LayerNorm(hdim))    
            
    def forward(self, q: Tensor, k_neigh: Tensor, v_neigh: Tensor, 
                geo_f: Tensor):
        '''
        Mememory efficient attention block.
        Input:
            config: Dict which contains parameters
            points: [B, n, 3]
            q: [B, n, n_head*hdim]
            k: [B, n, k, n_head*hdim]  
            v: [B, n, k, n_head*hdim]  
            geo_f: [B, n, k, n_head*hdim]
        '''
        B, N, nsample, _ = k_neigh.shape
        if q.shape[1] > self.q_bucket_size:
            self.q_bucket_size = q.shape[1]

        scale = q.shape[-1]**-0.5

        qk_weight = q.unsqueeze(dim=2)-k_neigh + geo_f               # [B, n, nsample, n_head*hdim]
        total_weight = (qk_weight)/scale                             # [B, n, nsample, n_head*hdim]
        total_weight = self.linear_weight(total_weight)

        # numerical stability
        # weight_max = total_weight.amax(dim = 2, keepdim = True).detach()      # max in nsample dim
        # total_weight = total_weight - weight_max
        exp_weight = torch.softmax(total_weight, dim=2)                         # [B, n, nsample, hdim]
        exp_weight = F.dropout(exp_weight, p = self.dropout)
        v_neigh = v_neigh.view(B, N, nsample, self.n_head, self.hdim)
        out = torch.sum(exp_weight.unsqueeze(dim=3)*v_neigh, dim=2)  # [B, n, n_head, hdim]

        return out


class Geoneigh_FPT_v2(nn.Module):
    def __init__(self, nsmaple, n_head, hdim, config):
        super().__init__()
        self.n_head = n_head
        self.hdim = hdim
        self.q_bucket_size = config.q_bucket_size
        self.eps = config.eps
        self.dropout = config.dropout
        self.weight_kernal = config.weight_kernal
        self.linear_weight = nn.Sequential(nn.Linear(int(hdim), hdim),
                                           nn.ReLU(),
                                           nn.LayerNorm(hdim))    
            
    def forward(self, q: Tensor, k_neigh: Tensor, v_neigh: Tensor, 
                geo_f: Tensor):
        '''
        Mememory efficient attention block.
        Input:
            config: Dict which contains parameters
            points: [B, n, 3]
            q: [B, N, n_head, hdim]
            k_neigh: [B, N, k, n_head, hdim]  
            v_neigh: [B, N, k, n_head, hdim]  
            geo_f: [B, N, k, hdim]
        '''
        # B, N, _, _, _ = k_neigh.shape
        if q.shape[1] > self.q_bucket_size:
            self.q_bucket_size = q.shape[1]

        scale = q.shape[-1]**-0.5
        q_chunks = q.split(self.q_bucket_size, dim = 1)
        k_chunks = k_neigh.split(self.q_bucket_size, dim = 1)
        v_chunks = v_neigh.split(self.q_bucket_size, dim = 1)
        geo_f_chunks = geo_f.split(self.q_bucket_size, dim = 1)
        
        # loop through all chunks
        out = []
        for _, (q_chunk, k_chunk, v_chunk, geo_f_chunk) in enumerate(zip(q_chunks, k_chunks, v_chunks ,geo_f_chunks)):
            qk_weight = q_chunk.unsqueeze(dim=2)-k_chunk+geo_f_chunk.unsqueeze(dim=3)   # [B, n, nsample, n_head, hdim]
            total_weight = (qk_weight)/scale                                            # [B, n, nsample, n_head, hdim]
            total_weight = self.linear_weight(total_weight)

            # numerical stability
            weight_max = total_weight.amax(dim = 2, keepdim = True).detach()        # max in nsample dim
            total_weight = total_weight - weight_max
            exp_weight = torch.softmax(total_weight, dim=2)                         # [B, n, nsample, n_head, hdim]
            exp_weight = F.dropout(exp_weight, p = self.dropout)
            weighted_value = torch.sum(exp_weight*v_chunk, dim=2)  # [B, n, n_head, hdim]
            out.append(weighted_value)
        out = torch.cat(out, dim = 1)
        return out

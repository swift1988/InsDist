import warnings

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class FRS(nn.Module):
    def __init__(self):
        super(FRS, self).__init__()

    
    def dist2(self, tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
   
        diff = (tensor_a - tensor_b) ** 2
        diff = diff * attention_mask
        diff = diff * channel_attention_mask
        diff = torch.sum(diff) ** 0.5

        return diff
        
    def forward(self, t_feats, x, stu_feature_adap):
        
        t = 0.1
        s_ratio = 1.0
        kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0

        #   for channel attention
        c_t = 0.1
        c_s_ratio = 1.0

        for _i in range(len(t_feats)):
            # spatial-teacher
            t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [1], keepdim=True)
            size = t_attention_mask.size()
            t_attention_mask = t_attention_mask.view(x[0].size(0), -1)
            t_attention_mask = torch.softmax(t_attention_mask / t, dim=1) * size[-1] * size[-2]
            t_attention_mask = t_attention_mask.view(size)
            # spatial-student
            s_attention_mask = torch.mean(torch.abs(x[_i]), [1], keepdim=True)
            size = s_attention_mask.size()
            s_attention_mask = s_attention_mask.view(x[0].size(0), -1)
            s_attention_mask = torch.softmax(s_attention_mask / t, dim=1) * size[-1] * size[-2]
            s_attention_mask = s_attention_mask.view(size)
            # channel-teacher
            c_t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
            c_size = c_t_attention_mask.size()
            c_t_attention_mask = c_t_attention_mask.view(x[0].size(0), -1)  # 2 x 256
            c_t_attention_mask = torch.softmax(c_t_attention_mask / c_t, dim=1) * 256
            c_t_attention_mask = c_t_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1
            # channel-student
            c_s_attention_mask = torch.mean(torch.abs(x[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
            c_size = c_s_attention_mask.size()
            c_s_attention_mask = c_s_attention_mask.view(x[0].size(0), -1)  # 2 x 256
            c_s_attention_mask = torch.softmax(c_s_attention_mask / c_t, dim=1) * 256
            c_s_attention_mask = c_s_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1
            # mask for feature imitation
            sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
            sum_attention_mask = sum_attention_mask.detach()
            c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
            c_sum_attention_mask = c_sum_attention_mask.detach()
            # feature imitation loss
            kd_feat_loss += self.dist2(t_feats[_i], stu_feature_adap[_i](x[_i]), attention_mask=sum_attention_mask, channel_attention_mask=c_sum_attention_mask) * 7e-5 * 6
            
           
        return kd_feat_loss

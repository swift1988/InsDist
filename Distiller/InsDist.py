import warnings

import torch
import torch.nn as nn
import torch.nn as nn
import cv2 as cv
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import imagebuild_backbone
import seaborn as sns
import os 
import xml.dom.minidom as xml
from mmcv.cnn import constant_init, kaiming_init



class InsDist(nn.Module):
    def __init__(self, weight_gt, weight_bg, threshold):
        super(InsDist, self).__init__()
        self.weight_gt = weight_gt
        self.weight_bg = weight_bg
        self.threshold = threshold
    
    
def mask_ours(self, gt_bboxes, backbone_feat, featmap_size, featmap_stride, threshold):
    
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    with torch.no_grad():
        mask_batch = []
        for batch in range(len(gt_bboxes)):
            
            h, w = featmap_size[0], featmap_size[1]
            mask_per_img = torch.zeros([h, w], dtype=torch.double).cuda()
            
            for ins in range(gt_bboxes[batch].shape[0]):
                gt_level_map = gt_bboxes[batch][ins] / featmap_stride
                
                lx = min(max(0, int(gt_level_map[0])), w - 1)
                rx = min(max(0, int(gt_level_map[2])), w - 1)
                ly = min(max(0, int(gt_level_map[1])), h - 1)
                ry = min(max(0, int(gt_level_map[3])), h - 1)
                
                if (lx == rx) or (ly == ry):
                    mask_per_img[ly, lx] += 1
                else:
                    x = backbone_feat[batch].view(-1, h * w).permute(1, 0)
                    feature_gt = avgpool(backbone_feat[batch][:, ly:(ry + 1), lx:(rx + 1)]).squeeze(-1)
                    energy = torch.mm(x, feature_gt)
                    
                    min_ = torch.min(energy)
                    max_ = torch.max(energy)
                    assert max_ != 0 
                    energy = (energy - min_) / max_
                    attention = energy.view(h, w)
                    
                    attention = (attention > threshold).double()
                    mask_per_img += attention
            mask_per_img = (mask_per_img > 0).double()
            mask_batch.append(mask_per_img)
            
    return torch.stack(mask_batch, dim=0)
    
    def dist(self, tensor_a, tensor_b, mask):
        diff = (tensor_a - tensor_b) ** 2
    
        mask_gt = mask.unsqueeze(1).repeat(1, tensor_a.size(1), 1, 1).cuda()
        diff_gt = diff * mask_gt
        diff_gt = (torch.sum(diff_gt) + 1e-8) ** 0.5
        
        mask_bg = (1 - mask_gt)
        diff_bg = diff * mask_bg
        diff_bg = (torch.sum(diff_bg) + 1e-8) ** 0.5
        
        return diff_gt, diff_bg
        
    def forward(self, feat_s, feat_t, gt_bbox, adaptation_layers):
        
         _mask = self.mask(gt_bbox, feat_t[-1], featmap_size=feat_t[-1].shape[2:], featmap_stride = 32, threshold=self.threshold).unsqueeze(1)
        feat_loss = 0
        
        for i in range(0, len(feat_t)):
            d_size = feat_t[i].shape[2:]
            mask = F.interpolate(_mask, d_size).squeeze(1)
            loss_gt, loss_bg = dist(feat_t[i], adaptation_layers[i](feat_s[i]), mask)
            feat_loss += (loss_gt * self.weight_gt + loss_bg * self.weight_bg)
            
        return feat_loss 
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



class DeFeat(nn.Module):
    def __init__(self, weight_gt, weight_bg):
        super(DeFeat, self).__init__()
        self.weight_gt = weight_gt
        self.weight_bg = weight_bg
    
    def gt_mask(self, gt_bboxes, featmap_size, featmap_stride):
        
        with torch.no_grad():
            mask_batch = []
            for batch in range(len(gt_bboxes)):
                h, w = featmap_size[0], featmap_size[1]
                mask_per_img = torch.zeros([h, w], dtype=torch.double).cuda()
                for ins in range(gt_bboxes[batch].shape[0]):
                    gt_level_map = gt_bboxes[batch][ins] / featmap_stride
                    lx = int(gt_level_map[0])
                    lx = min(lx, w - 1)
                    rx = int(gt_level_map[2])
                    rx = min(rx, w - 1)
                    
                    ly = int(gt_level_map[1])
                    ly = min(ly, h - 1)
                    ry = int(gt_level_map[3])
                    ry = min(ry, h - 1)
                    
                    if (lx == rx) or (ly == ry):
                        mask_per_img[ly, lx] += 1
                    else:
                        mask_per_img[ly:ry, lx:rx] += 1
                
                mask_per_img = (mask_per_img > 0).double()
                
                mask_batch.append(mask_per_img)
            
        return torch.stack(mask_batch, dim=0)
    
    def defeat(self, tensor_a, tensor_b, mask):
        diff = (tensor_a - tensor_b) ** 2
    
        mask_gt = mask.unsqueeze(1).repeat(1, tensor_a.size(1), 1, 1).cuda()
        diff_gt = diff * mask_gt
        diff_gt = (torch.sum(diff_gt) + 1e-8) ** 0.5
        
        mask_bg = (1 - mask_gt)
        diff_bg = diff * mask_bg
        diff_bg = (torch.sum(diff_bg) + 1e-8) ** 0.5
        
        return diff_gt, diff_bg
        
    def forward(self, feat_s, feat_t, gt_bbox, adaptation_layers):
        strides = [8, 16, 32, 64, 128]
        feat_loss = 0
        
        for i in range(0, len(feat_t)):
            
            featmap_size = feat_t[i].shape[2:]
            
            mask = self.gt_mask(gt_bboxes, featmap_size, strides[i])
            
            loss_gt, loss_bg = self.dist2_mask(feat_t[i], adaptation_layers[i](neck_feat_s[i]), mask)
            feat_loss += (self.weight_gt * loss_gt + self.weight_bg * loss_bg)
            
        return feat_loss 
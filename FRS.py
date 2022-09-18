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


class FRS(nn.Module):
    def __init__(self):
        super(FRS, self).__init__()

    def forward(self, x, y, tea_bbox_outs, stu_feature_adap):
        
        tea_cls_score = tea_bbox_outs[0]

        layers = len(tea_cls_score)
        
        distill_feat_loss, distill_cls_loss = 0, 0

        for layer in range(layers):
            
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values
            mask = mask.detach()

            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer](x[layer])), 2)

            loss = (feat_loss * mask[:,None,:,:]).sum()
            if loss > 1000000:
                loss = 0
            
            distill_feat_loss += loss / mask.sum()
                    
            # print((feat_loss * mask[:,None,:,:]).sum(), mask.sum())
        return distill_feat_loss        
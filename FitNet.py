import torch
import torch.nn as nn
import torch.nn.functional as F




class FitNet(nn.Module):
    def __init__(self, s_channels, t_channels):
        super(FitNet, self).__init__()
        
        # adaptation
        self.conv = nn.Conv2d(s_channels, t_channels, kernel_size=1, bias=False)

    def forward(self, feat_S, feat_T):
        B, C, H, W = feat_S[0].size()

        distill_feat_loss = 0
        for i in range(len(feat_S)):

            feat_S = self.conv(feat_S)

            distill_feat_loss += ((feat_S - feat_T)**2)
        
        return distill_feat_loss / len(feat_S)

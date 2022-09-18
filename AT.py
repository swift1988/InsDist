import torch
import torch.nn as nn
import torch.nn.functional as F




class AT(nn.Module):
    def __init__(self):
        super(AT, self).__init__()
        self.p = 2

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

    def forward(self, feat_S, feat_T):
        for i range(0, len(feat_S)):
            
            loss += (self.at(feat_S[i]) - self.at(feat_T[i])).pow(2).mean()
            
        return loss / len(feat_S)

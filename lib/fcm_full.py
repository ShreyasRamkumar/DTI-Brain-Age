from torch import optim, nn, sigmoid, cat
from torch.nn.functional import relu
from .network_utility import *
from swin_transformer import SwinTransformerBlock
from numpy import multiply


class FeatureComplementaryModule(nn.Module):
    def __init__(self, h:int, w:int, c:int):
        super().__init__()
        self.h, self.w, self.c = h, w, c
        self.cfb = self.CrossDomainFusionBlock(h, w, c)
        self.ceb = self.CorrelationEnhancementBlock(h, w, c)
        self.cab = self.ChannelAttentionBlock(h, w, c)
        self.ffb = self.FeatureFusionBlock(h, w, c)

    def forward(self, gi, fi):
        si = self.cfb(gi, fi)
        ei = self.ceb(gi, fi)
        ai = self.cab(gi)
        mi = self.ffb(si, ei, ai)
        return mi
    

    class CrossDomainFusionBlock(nn.Module):
        def __init__(self, h, w, c):
            super().__init__()
            self.h, self.w, self.c = h, w, c
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.stb = SwinTransformerBlock(c)
            self.conv = nn.Conv2d(2*c, c, kernel_size=1)

        def forward(self, gi, fi):
            gi = gi.view(1, self.c, self.h, self.w)
            fi = fi.view(1, self.c, self.h, self.w)
            
            gi_pool = self.gap(gi).view(1, self.c)
            fi_pool = self.gap(fi).view(1, self.c)
            
            gi1 = cat([gi.view(self.h*self.w, self.c), fi_pool], dim=0)
            fi1 = cat([fi.view(self.h*self.w, self.c), gi_pool], dim=0)
            
            gi3 = self.stb(gi1).view(1, self.c, self.h, self.w)
            fi3 = self.stb(fi1).view(1, self.c, self.h, self.w)
            
            si = self.conv(cat([gi3, fi3], dim=1))
            return si.squeeze(0).permute(1, 2, 0)  # Return as (h, w, c)

    class CorrelationEnhancementBlock(nn.Module):
        def __init__(self, h, w, c):
            super().__init__()
            self.h, self.w, self.c = h, w, c

        def forward(self, gi, fi):
            gi0 = gi.view(self.h, self.w, self.c)
            ei = multiply(gi0, fi)
            return ei

    class ChannelAttentionBlock(nn.Module):
        def __init__(self, h, w, c):
            super().__init__()
            self.h, self.w, self.c = h, w, c
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(c, c // 2)
            self.fc2 = nn.Linear(c // 2, c)

        def forward(self, gi):
            gi = gi.view(1, self.c, self.h, self.w)
            y = self.gap(gi).view(self.c)
            y = relu(self.fc1(y))
            y = sigmoid(self.fc2(y)).view(1, self.c, 1, 1)
            return (gi * y).squeeze(0).permute(1, 2, 0)  # Return as (h, w, c)

    class FeatureFusionBlock(nn.Module):
        def __init__(self, h, w, c):
            super().__init__()
            self.h, self.w, self.c = h, w, c
            self.conv1 = nn.Conv2d(3*c, c, kernel_size=1)
            self.cbr = nn.Sequential(
                nn.Conv2d(3*c, c, kernel_size=1),
                nn.BatchNorm2d(c),
                nn.ReLU()
            )

        def forward(self, si, ei, ai):
            mi1 = cat([si, ei, ai], dim=2)
            mi1 = mi1.permute(2, 0, 1).unsqueeze(0)  # (1, 3c, h, w)
            out1 = self.conv1(mi1)
            out2 = self.cbr(mi1)
            mi = out1 + out2
            return mi.squeeze(0).permute(1, 2, 0)  # Return as (h, w, c)
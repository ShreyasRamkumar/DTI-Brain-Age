from torch import optim, nn, sigmoid, cat
from torch.nn.functional import relu
from .network_utility import *
from numpy import array
from swin_transformer import *
from fcm_full import FeatureComplementaryModule


class Hybrid3_Utility:
    def __init__(self):
        self.f = []
        self.c = None # convolution block result
        self.t = None # transformer block result


class StandardFusion(nn.Module, Hybrid3_Utility):
    def __init__(self):
        super.__init__()

    def layer(self, x: array, count: int = 0, dim: int = 1, in_c: int = 1, out_c: int = 48, 
              input_resolution: tuple[int] = (32, 32), stride: int = 2):
        if count == 1:
            conv1x = self.conv(x=x, in_c=in_c, out_c=out_c, stride=stride)
            conv2x = self.conv(x=conv1x, in_c=out_c, out_c=out_c, stride=stride)
            self.c = conv2x
        else:
            convx = self.conv(x=x, in_c=in_c, out_c=out_c, stride=stride)
            self.c = convx

        trans1x = self.transform(x=x, dim=dim, heads=3, input_resolution=input_resolution, depth=2)
        self.t = trans1x
        shape = self.c.shape
        fcm = FeatureComplementaryModule(h = shape[0], w = shape[1], c = shape[2])
        fusionx = fcm.forward(gi=self.t, fi=self.c)
        self.f.append(fusionx)

    def compile_run(self):
        return nn.Sequential(*self.layers)
    
    def conv(self, x: array, in_c: int, out_c: int, stride = 1):
        convolution = nn.Conv2d(in_channels=in_c, out_channels=out_c, stride=stride)
        return convolution.forward(x)
    
    def transform(self, x: array, dim: int, heads: int, input_resolution: tuple[int], depth: int, use_checkpoint: bool = False):
        transformer = BasicLayer(dim=dim, num_heads=heads, input_resolution=input_resolution, depth=depth, use_checkpoint=use_checkpoint)
        return transformer.forward(x)

    

            




            
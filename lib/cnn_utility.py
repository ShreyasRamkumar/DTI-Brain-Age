from torch import optim, nn, ones, sqrt, mean, std, abs
import torch.nn.functional as F
from tqdm import tqdm
import skimage

class network_utility:
    @staticmethod
    def convolution(in_c, out_c):
        run = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_c),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return run
    
    @staticmethod
    def fcn_layers():
        run = nn.Sequential(
            nn.Linear(256*2*2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        return run
    
    
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
        return FCNLayers()

class FCNLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
from torch import optim, nn, ones, sqrt, mean, std, abs
import torch.nn.functional as F
from tqdm import tqdm
import skimage

class Network_Utility:
    @staticmethod
    def convolution(in_c, out_c):
        run = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_c),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return run
    
    @staticmethod
    def fcn_layers():
        run = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    @staticmethod
    def create_data_splits(dataset_len):
        training_len = int(dataset_len * 0.8)
        validation_len = int((dataset_len - training_len) / 2)
        return [training_len, validation_len, validation_len]
    
    @staticmethod
    def get_slice(scan_tensor):
        scan_entropies = []
        for i in tqdm(range(192)):
            scan_slice = scan_tensor[:, :, i]
            entropy = skimage.measure.shannon_entropy(scan_slice)
            scan_entropies.append(entropy)
        max_entropy = max(scan_entropies)
        max_entropy_slice_index = scan_entropies.index(max_entropy)
        return max_entropy_slice_index
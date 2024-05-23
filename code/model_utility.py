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
            nn.BatchNorm2d(num_features=out_c)
        )
        return run
    
    @staticmethod
    def down_convolution(in_c, out_c):
        run = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_c)
        )
        return run

    @staticmethod
    def up_convolution(in_c, out_c):
        run = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_c),
        )
        return run

    @staticmethod
    def final_convolution(in_c, out_c):
        run = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1),
            nn.ReLU()
        )
        return run

    @staticmethod
    def crop_tensor(target_tensor, tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2

        return tensor[:, :, delta:tensor_size- delta, delta:tensor_size-delta]
    
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
import skimage
from torch import nn

class network_utility:
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

# taken from Papers With Code (https://paperswithcode.com/method/inverted-residual-block)
class InvertedResidual(nn.module):
     def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

# taken from Papers With Code (https://paperswithcode.com/method/inverted-residual-block)
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class FFN(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        # Add residual and normalize
        out = out + residual
        out = self.layer_norm(out)
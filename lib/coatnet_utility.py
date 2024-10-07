import torch.nn as nn
import torch.nn.functional as F

# Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

# Depthwise Separable Convolution Block
class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion=4):
        super(DepthwiseConvBlock, self).__init__()
        expanded_channels = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, expanded_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size,
                                   stride=stride, groups=expanded_channels, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(expanded_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.bn2 = nn.BatchNorm2d(expanded_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.depthwise(x)))
        x = self.bn3(self.conv2(x))
        return x

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super(ResidualBlock, self).__init__()
        self.depthwise_block = DepthwiseConvBlock(in_channels, out_channels, stride=stride, expansion=expansion)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.depthwise_block(x)
        return F.relu(x + residual)

# Attention Block (Basic Self-Attention)
class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super(AttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)

    def forward(self, x):
        # x has shape [batch_size, channels, height, width]
        batch_size, channels, height, width = x.shape

        # Flatten the spatial dimensions (height and width) into a sequence
        x = x.view(batch_size, channels, height * width)  # [batch_size, channels, sequence_length]

        # Transpose to match the input format for MultiheadAttention
        x = x.permute(2, 0, 1)  # [sequence_length, batch_size, channels]

        # Apply LayerNorm and MultiheadAttention
        x = self.norm(x)
        attn_output, _ = self.attention(x, x, x)

        # Transpose back to the original format
        x = attn_output.permute(1, 2, 0)  # [batch_size, channels, sequence_length]

        # Reshape to original spatial dimensions
        x = x.view(batch_size, channels, height, width)  # [batch_size, channels, height, width]

        return x
import csv
from typing import List

import skimage
import torch
from torch import nn, einsum, sqrt, FloatTensor, arange
from tqdm import tqdm

class Nu:
    @staticmethod
    def create_data_splits(dataset_len):
        training_len = int(dataset_len * 0.8)
        validation_len = int((dataset_len - training_len) / 2)
        return [training_len, validation_len, validation_len]
    
    @staticmethod
    def get_slice(scan_tensor):
        scan_entropies = []
        for i in tqdm(range(68)):
            scan_slice = scan_tensor[:, :, i]
            entropy = skimage.measure.shannon_entropy(scan_slice)
            scan_entropies.append(entropy)
        max_entropy = max(scan_entropies)
        max_entropy_slice_index = scan_entropies.index(max_entropy)
        return max_entropy_slice_index

    @staticmethod
    def access_encoder_wb(b) -> List[torch.Tensor]:
        attention = b.attention
        intermediate_weight = b.intermediate.dense.weight
        intermediate_bias = b.intermediate.dense.bias
        layernorm_before_weight = b.layernorm_before.weight
        layernorm_before_bias = b.layernorm_before.bias
        layernorm_after_weight = b.layernorm_after.weight
        layernorm_after_bias = b.layernorm_after.bias
        qkv_weights = [attention.self.query.weight, attention.self.key.weight, attention.self.value.weight]
        qkv_bias = [attention.self.query.bias, attention.self.key.bias, attention.self.value.bias]
        output_weight = b.output.dense.weight
        output_bias = b.output.dense.bias

        return [intermediate_weight, intermediate_bias, layernorm_before_weight, layernorm_before_bias,
                layernorm_after_weight, layernorm_after_bias, qkv_weights, qkv_bias, output_weight, output_bias]

    @staticmethod
    def set_encoder_wb(c_swin, l_idx, b_idx, w_b):
        c_swin.encoder.layers[l_idx].blocks[b_idx].attention.self.query.weight.data = w_b[6][0]
        c_swin.encoder.layers[l_idx].blocks[b_idx].attention.self.key.weight.data = w_b[6][1]
        c_swin.encoder.layers[l_idx].blocks[b_idx].attention.self.value.weight.data = w_b[6][2]

        c_swin.encoder.layers[l_idx].blocks[b_idx].attention.self.query.bias.data = w_b[7][0]
        c_swin.encoder.layers[l_idx].blocks[b_idx].attention.self.key.bias.data = w_b[7][1]
        c_swin.encoder.layers[l_idx].blocks[b_idx].attention.self.value.bias.data = w_b[7][2]

        c_swin.encoder.layers[l_idx].blocks[b_idx].intermediate.dense.weight.data = w_b[0]
        c_swin.encoder.layers[l_idx].blocks[b_idx].intermediate.dense.bias.data = w_b[1]

        c_swin.encoder.layers[l_idx].blocks[b_idx].layernorm_before.weight.data = w_b[2]
        c_swin.encoder.layers[l_idx].blocks[b_idx].layernorm_before.bias.data = w_b[3]

        c_swin.encoder.layers[l_idx].blocks[b_idx].layernorm_after.weight.data = w_b[4]
        c_swin.encoder.layers[l_idx].blocks[b_idx].layernorm_after.bias.data = w_b[5]

        c_swin.encoder.layers[l_idx].blocks[b_idx].output.dense.weight.data = w_b[8]
        c_swin.encoder.layers[l_idx].blocks[b_idx].output.dense.bias.data = w_b[9]

    @staticmethod
    def import_data(csv_file, scan_paths):
        ages = []
        scan_counter = 0

        with open(csv_file, mode='r') as file:
            csv_reader = csv.reader(file)

            for i, row in enumerate(csv_reader):
                if row[0] in scan_paths[scan_counter]:
                    scan_counter += 1
                    age = row[1]
                    ages.append(age)

        return ages

# taken from Papers With Code (https://paperswithcode.com/method/inverted-residual-block)
class InvertedResidual(nn.Module):
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

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# taken from Papers With Code (https://paperswithcode.com/method/inverted-residual-block)
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            # norm_layer(out_planes).eval(),
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

class RelativeAttention(nn.Module):
    def __init__(self, embed_size, num_heads, max_len=5000, dropout=0.1):
        super(RelativeAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.scale = sqrt(FloatTensor([embed_size // num_heads]))

        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.relative_positions_embeddings = nn.Embedding(2 * max_len - 1, embed_size)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_length, embed_size = query.size()
        
        # Generate relative position indices
        relative_positions_matrix = self._generate_relative_positions_matrix(seq_length)
        relative_positions_matrix = relative_positions_matrix.to(query.device)
        relative_positions_embeddings = self.relative_positions_embeddings(relative_positions_matrix)
        
        # Compute attention with relative position embeddings
        attention_output, attention_weights = self.attention(query, key, value, attn_mask=mask)
        attention_output += self._relative_attention(query, key, relative_positions_embeddings)
        
        return self.dropout(attention_output), attention_weights

    def _relative_attention(self, query, key, relative_positions_embeddings):
        seq_length = query.size(1)
        query = query / self.scale

        # Compute relative attention scores
        relative_attention_scores = einsum('bhqd,ld->bhql', query, relative_positions_embeddings)
        relative_attention_scores = relative_attention_scores / self.scale

        return relative_attention_scores

    def _generate_relative_positions_matrix(self, length):
        range_vec = arange(length)
        range_matrix = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        relative_positions_matrix = range_matrix + self.max_len - 1
        return relative_positions_matrix
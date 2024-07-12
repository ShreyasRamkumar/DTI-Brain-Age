from torch import optim, nn, einsum, arange, sqrt, FloatTensor
import torch.nn.functional as F
from network_utility import *

class Hybrid1_Utility:
    @staticmethod
    def s0():
        run = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.Conv2d(in_channels=1, out_channels=64, stride=2, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=3),
            nn.GELU()
        )
        return run
    
    @staticmethod
    def s1():
        layers = [nn.BatchNorm2d(num_features=128)]
        conv = nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3)
        dconv = InvertedResidual(inp=128, oup=128, stride=1, expand_ratio=4)
        for i in range(6):
            layers.append(conv)
            layers.append(dconv)
            layers.append(conv)
        layers.append(nn.GELU())
        run = nn.Sequential(*layers)
        return run
    
    @staticmethod
    def s2():
        layers = [nn.BatchNorm2d(num_features=128)]
        conv1 = nn.Conv2d(in_channels=128, out_channels=256, stride=2, kernel_size=3)
        conv = nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=3)
        dconv = InvertedResidual(inp=256, oup=256, stride=1, expand_ratio=4)
        for i in range(6):
            if i == 0:
                layers.append(conv1)
                layers.append(dconv)
                layers.append(conv)
            else:
                layers.append(conv)
                layers.append(dconv)
                layers.append(conv)
        layers.append(nn.GELU())
        run = nn.Sequential(*layers)
        return run
    
    @staticmethod
    def s3():
        ra = RelativeAttention(embed_size=512, num_heads=16)
        ffn = FFN(input_dim=512, hidden_dim=1024)
        layers = [nn.LayerNorm([512, 28, 28])]
        for i in range(14):
            layers.append(ra)
            layers.append(ffn)
        layers.append(nn.GELU())
        run = nn.Sequential(*layers)
        return run
    
    @staticmethod
    def s4():
        ra = RelativeAttention(embed_size=1024, num_heads=32)
        ffn = FFN(input_dim=1024, hidden_dim=4096)
        layers = [nn.LayerNorm([1024, 14, 14])]
        for i in range(2):
            layers.append(ra)
            layers.append(ffn)
        run = nn.Sequential(*layers)
        return run
    
    @staticmethod
    def gbp():
        run = nn.Sequential(nn.AvgPool2d(kernel_size=7, stride=1, padding=0))
        return run

    @staticmethod
    def fc():
        pass

class FCN(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(x.size(0) - 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
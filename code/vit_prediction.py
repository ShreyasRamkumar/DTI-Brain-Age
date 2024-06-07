import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl

# Define the patch size and number of patches
patch_size = 16
num_patches = (128 // patch_size) ** 2

# Define the transformer hyperparameters
num_heads = 8
num_layers = 6
dim_feedforward = 2048
dropout = 0.1

# Define the image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define the patch embedding layer
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# Define the transformer encoder layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.self_attn(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.feedforward(x)
        x = self.norm2(x)
        return x
    
class VisionTransformer(pl.LightningModule):
    def __init__(self, in_channels, patch_size, embed_dim, num_heads, num_layers, dim_feedforward, dropout, num_classes):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x[:, 0]
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    # Define the model parameters
    in_channels = 3
    embed_dim = 768
    num_classes = 1

    # Create the model
    model = VisionTransformer(in_channels, patch_size, embed_dim, num_heads, num_layers, dim_feedforward, dropout, num_classes)

    # Example usage
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1)
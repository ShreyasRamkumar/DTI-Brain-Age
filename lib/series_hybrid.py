from transformers import SwinConfig, SwinModel
from network_utility import network_utility as nu
from network_utility import InvertedResidual
import torch
import torch.nn as nn

class SeriesHybrid(nn.Module):
    def __init__(self):
        super(SeriesHybrid, self).__init__()
        # INITIALIZING TRANSFORMER
        config = SwinConfig.from_pretrained("microsoft/swin-base-patch4-window7-224")

        custom_config = SwinConfig(
            image_size=128,
            patch_size=4,
            num_channels=1,
            embed_dim=128,
            num_heads=[4, 8, 16, 32],
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-5,
            window_size=4
        )

        pretrained_swin = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224", config=config)

        custom_swin = SwinModel(custom_config)

        layer_idx = 0
        for layer in pretrained_swin.encoder.layers:
            if layer_idx != 2:
                block_idx = 0
                for block in layer.blocks:
                    wb = nu.access_encoder_wb(block)

                    nu.set_encoder_wb(custom_swin, layer_idx, block_idx, wb)

                    block_idx += 1

            else:
                for i in range(0, 15, 3):
                    wb1 = nu.access_encoder_wb(layer.blocks[i])
                    wb2 = nu.access_encoder_wb(layer.blocks[i + 1])
                    wb3 = nu.access_encoder_wb(layer.blocks[i + 2])

                    avg_wb = []

                    for j in range(len(wb1)):
                        if isinstance(wb1[j], torch.FloatTensor):
                            avg_wb.append((wb1[j] + wb2[j] + wb3[j]) / 3)

                        else:
                            q = (wb1[j][0] + wb2[j][0] + wb3[j][0]) / 3
                            k = (wb1[j][1] + wb2[j][1] + wb3[j][1]) / 3
                            v = (wb1[j][2] + wb2[j][2] + wb3[j][2]) / 3
                            avg_wb.append([q, k, v])

                    target_index = i // 3

                    nu.set_encoder_wb(custom_swin, layer_idx, target_index, avg_wb)

            if layer_idx != 3:
                reduction_weight = layer.downsample.reduction.weight

                custom_swin.encoder.layers[layer_idx].downsample.reduction.weight.data = reduction_weight

            layer_idx += 1

        self.a1 = custom_swin.encoder.layers[1]
        self.a2 = custom_swin.encoder.layers[2]
        self.avgp = nn.AdaptiveAvgPool1d(1)
        self.regression_head = nn.Linear(512, 1)

    # STAGE DEFINITIONS
    def s0(self, x):
        run = nn.ModuleList(
            [nn.BatchNorm2d(num_features=1),
             nn.Conv2d(in_channels=1, out_channels=64, stride=2, kernel_size=3),
             nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=3),
             nn.GELU()]
        )
        for block in run:
            x = block(x)
        return x

    def s1(self, x):
        layers = [nn.BatchNorm2d(num_features=128)]
        conv = nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3)
        dconv = InvertedResidual(inp=128, oup=128, stride=1, expand_ratio=4)
        for i in range(6):
            layers.append(conv)
            layers.append(dconv)
            layers.append(conv)
        layers.append(nn.GELU())
        for block in layers:
            x = block(x)
        return x


    def s2(self, x):
        layers = [nn.BatchNorm2d(num_features=128)]
        conv1 = nn.Conv2d(in_channels=128, out_channels=256, stride=2, kernel_size=1)
        conv = nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=1)
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
        for block in layers:
            x = block(x)
        return x

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)

        x = x.reshape(1, 16, 256)

        x = self.a1(x, (4,4))
        x = x[0]
        x = x.permute(0, 2, 1)
        x = self.avgp(x).squeeze(-1)
        output = self.regression_head(x)
        print(output)
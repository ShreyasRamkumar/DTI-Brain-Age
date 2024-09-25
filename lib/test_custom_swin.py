from typing import List
from network_utility import network_utility as nu
import torch
import torch.nn as nn
from transformers import SwinModel, SwinConfig
import numpy as np
from network_utility import network_utility
from Preprocessing import Preprocessing
import nibabel as nib

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

# LOADING SCAN
scan = nib.load("C:\\Code\\GPN\\DTI-Brain-Age\\testing\\fa_testing.nii")
scan_array = scan.get_fdata()
scan_tensor = torch.from_numpy(scan_array)
slice_int = nu.get_slice(scan_tensor)

slice_tensor = scan_tensor[:, :, slice_int].float()
slice_tensor = torch.unsqueeze(slice_tensor, 0)
slice_tensor = torch.unsqueeze(slice_tensor, 0)

padded_tensor = Preprocessing.pad_to_128(img_tensor=slice_tensor)
#
# PATCH EMBEDDING WEIGHT TRANSFER AND EXECUTION
pretrained_pe_weights = pretrained_swin.embeddings.patch_embeddings.projection.weight.detach().cpu().numpy()
pretrained_pe_bias = pretrained_swin.embeddings.patch_embeddings.projection.bias.detach().cpu().numpy()

adapted_pe_weights = pretrained_pe_weights.mean(axis=1, keepdims=True)

custom_swin.embeddings.patch_embeddings.projection.weight.data = torch.from_numpy(adapted_pe_weights)
custom_swin.embeddings.patch_embeddings.projection.bias.data = torch.from_numpy(pretrained_pe_bias)

pe = custom_swin.embeddings
x, i_d = pe.forward(padded_tensor)

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
            wb2 = nu.access_encoder_wb(layer.blocks[i+1])
            wb3 = nu.access_encoder_wb(layer.blocks[i+2])

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


regression_head = nn.Linear(custom_swin.config.hidden_size, 1)

for layer in custom_swin.encoder.layers:
    x = layer.forward(x, i_d)
    i_d = x[2][2:]
    x = x[0]
pooled_output = x.mean(dim=1)
output = regression_head(pooled_output)
print(output)

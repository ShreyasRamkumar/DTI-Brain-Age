import pytorch_lightning as pl
from cnn_utility import network_utility

# Model Class
class CNN(pl.LightningDataModule):
    def __init__(self, learning_rate: int = 1e-3):

        # definition of neural network (naming convention = o_number of channels_encode/decode_up/down/side)
        super().__init__()
        self.o_1 = network_utility.convolution(1, 16)
        self.o_2 = network_utility.convolution(16, 32)
        self.o_3 = network_utility.convolution(32, 64)
        self.o_4 = network_utility.convolution(64, 128)
        self.o_5 = network_utility.convolution(128, 256)
        self.o_6 = network_utility.fcn_layers()

    # forward pass
    def forward(self, image):
        # naming convention: x_number of channels_encode/decode_up/down/nothing(side convolution)

        conv_1 = self.o_1(image)
        conv_2 = self.o_2(conv_1)
        conv_3 = self.o_3(conv_2)
        conv_4 = self.o_4(conv_3)
        conv_5 = self.o_5(conv_4)
        conv_5_flat = conv_5.view(conv_5.size(0), -1)
        age_prediction = self.o_6(conv_5_flat)
        return age_prediction.item()
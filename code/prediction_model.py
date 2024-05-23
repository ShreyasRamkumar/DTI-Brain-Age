import os
import lightning.pytorch as pl
from torch import optim, nn
import torch

import nibabel as nib
import numpy as np
from lightning.pytorch.callbacks import Callback
from Network_Utility import Network_Utility

 
# important folders
x_directory = "/home/ramkumars@acct.upmchs.net/Projects/Harmonizing-MRI-Scans/data/processed_input/" # CHANGE FOR WHEN USING JENKINS

class Callbacks(Callback):
    def on_test_end(self, trainer, pl_module):
        outputs = pl_module.testing_outputs
        sliced_yhat_list = []
        
        for i in range(len(outputs)):
            yhat = outputs[i]
            for j in range(4):
                sliced_yhat = yhat[j:j+1, :, :, :]
                print(sliced_yhat.shape)
                if sliced_yhat.shape == torch.Size([0, 1, 256, 256]):
                    sliced_yhat = sliced_yhat.reshape(0, 256, 1, 256)
                else:
                    sliced_yhat = sliced_yhat.reshape(1, 256, 1, 256)
                sliced_yhat = sliced_yhat.squeeze(dim=0)
                print(sliced_yhat.shape)
                sliced_yhat_list.append(sliced_yhat)

        for i in range(len(sliced_yhat_list)):
            y_hat_path = f"{yhat_directory}/{i}.nii"
            sliced_y_hat_tensor = sliced_yhat_list[i].cpu()
            sliced_y_hat_array = sliced_y_hat_tensor.numpy()
            y_hat_scan = nib.Nifti1Image(sliced_y_hat_array, affine=np.eye(4))
            nib.save(y_hat_scan, y_hat_path)

# Model Class
class Unet(pl.LightningModule):
    def __init__(self,learning_rate=1e-3):
        super().__init__()
        # hyperparameters
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss
        self.testing_outputs = []
        self.validation_outputs = []
        
        # definition of neural network (naming convention = o_number of channels_encode/decode_up/down/side)
        self.o_16_encode_side = Network_Utility.convolution(1, 16)
        self.o_16_encode_down = Network_Utility.down_convolution(16, 16)
        self.o_32_encode_side = Network_Utility.convolution(16, 32)
        self.o_32_encode_down = Network_Utility.down_convolution(32, 32)
        self.o_64_encode_side = Network_Utility.convolution(32, 64)
        self.o_64_encode_down = Network_Utility.down_convolution(64, 64)
        self.o_128_encode_side = Network_Utility.convolution(64, 128)
        self.o_128_encode_down = Network_Utility.down_convolution(128, 128)
        self.o_256_encode_side = Network_Utility.convolution(128, 256)
        self.o_128_decode_up = Network_Utility.up_convolution(256, 128)
        self.o_128_decode_side = Network_Utility.convolution(256, 128)
        self.o_64_decode_up = Network_Utility.up_convolution(128, 64)
        self.o_64_decode_side = Network_Utility.convolution(128, 64)
        self.o_32_decode_up = Network_Utility.up_convolution(64, 32)
        self.o_32_decode_side = Network_Utility.convolution(64, 32)
        self.o_16_decode_up = Network_Utility.up_convolution(32, 16)
        self.o_16_decode_side = Network_Utility.convolution(32, 16)
        self.o_1_decode_side = Network_Utility.final_convolution(17, 1)
    
    # forward pass
    def forward(self, image):
        # naming convention: x_number of channels_encode/decode_up/down/nothing(side convolution)
        
        x_16_encode = self.o_16_encode_side(image) # has a shape of [1, 16, 256, 256]
        x_16_encode_down = self.o_16_encode_down(x_16_encode) # has a shape of [1, 16, 128, 128]
        x_32_encode = self.o_32_encode_side(x_16_encode_down) # has a shape of [1, 32, 128, 128]
        x_32_encode_down = self.o_32_encode_down(x_32_encode) # has a shape of [1, 32, 64, 64]
        x_64_encode = self.o_64_encode_side(x_32_encode_down) # has a shape of [1, 64, 64, 64]     
        x_64_encode_down = self.o_64_encode_down(x_64_encode) # has a shape of [1, 64, 32, 32]
        x_128_encode = self.o_128_encode_side(x_64_encode_down)  # has a shape of [1, 128, 32, 32]      
        x_128_encode_down = self.o_128_encode_down(x_128_encode) # has a shape of [1, 128, 16, 16]
        x_256_encode = self.o_256_encode_side(x_128_encode_down) # has a shape of [1, 256, 16, 16]

        x_256_decode_up = self.o_128_decode_up(x_256_encode) # has a shape of [1, 128, 22, 22]
        x_256_decode_cat = torch.cat([x_256_decode_up, x_128_encode], 1) # has a shape of [1, 256, 32, 32]       
        x_128_decode = self.o_128_decode_side(x_256_decode_cat) # has a shape of [1, 128, 32, 32]
        x_128_decode_up = self.o_64_decode_up(x_128_decode) # has a shape of [1, 64, 64, 64]
        x_128_decode_cat = torch.cat([x_128_decode_up, x_64_encode], 1) # has a shape of [1, 128, 64, 64]
        x_64_decode = self.o_64_decode_side(x_128_decode_cat) # has a shape of [1, 64, 64, 64]
        x_64_decode_up = self.o_32_decode_up(x_64_decode) # has a shape of [1, 32, 128, 128]
        x_64_decode_cat = torch.cat([x_64_decode_up, x_32_encode], 1) # has a shape of [1, 64, 128, 128]
        x_32_decode = self.o_32_decode_side(x_64_decode_cat) # has a shape of [1, 32, 128, 128]
        x_32_decode_up = self.o_16_decode_up(x_32_decode) # has a shape of [1, 16, 256, 256]
        x_32_decode_cat = torch.cat([x_32_decode_up, x_16_encode], 1) # has a shape of [1, 32, 256, 256]
        x_16_decode = self.o_16_decode_side(x_32_decode_cat) # has a shape of [1, 16, 256, 256]
        x_16_decode_cat = torch.cat([x_16_decode, image], 1) # has a shape of [1, 17, 256, 256]
        final_image = self.o_1_decode_side(x_16_decode_cat)

        assert final_image.shape == image.shape
        
        return final_image

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr = self.learning_rate)
        return opt
    
    def training_step(self, train_batch, batch_idx):
        x = train_batch["scan"]
        y = train_batch["ground_truth"]
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x = test_batch["scan"]
        y = test_batch["ground_truth"]
        y_hat = self.forward(x)
        self.testing_outputs.append(y_hat)

        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["scan"]
        y = val_batch["ground_truth"]
        y_hat = self.forward(x)

        self.validation_outputs.append(y_hat)

        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)
    

class MRIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.training = []
        self.ground_truth_training = []
        self.testing = []
        self.ground_truth_testing = []
        self.validation = []
        self.ground_truth_validation = []
        self.input_files = os.listdir(x_directory)
        self.ground_truth_files = os.listdir(y_directory)

    def setup(self, stage: str):
        # set up training, testing, validation split
        
        lens = Network_Utility.create_data_splits(len(self.input_files))
        training_stop_index = lens[0]
        testing_stop_index = lens[0] + lens[1]
        validation_stop_index = lens[0] + lens[1] + lens[2] - 1

        self.training = self.input_files[:training_stop_index]
        self.ground_truth_training = self.ground_truth_files[:training_stop_index]

        self.training_dataset = MRIDataset(self.training, self.ground_truth_training)
        self.training_dataloader = self.train_dataloader()

        self.testing = self.input_files[training_stop_index:testing_stop_index]
        self.ground_truth_testing = self.ground_truth_files[training_stop_index:testing_stop_index]  

        self.testing_dataset = MRIDataset(self.testing, self.ground_truth_testing)
        self.testing_dataloader = self.test_dataloader()

        self.validation = self.input_files[testing_stop_index:validation_stop_index] 
        self.ground_truth_validation = self.ground_truth_files[testing_stop_index:validation_stop_index]

        self.validation_dataset = MRIDataset(self.validation, self.ground_truth_validation)
        self.validation_dataloader = self.val_dataloader()


    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size = self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.testing_dataset, batch_size = self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size = self.batch_size)


if __name__ == "__main__":
    mri_data = MRIDataModule(batch_size=4)
    model = Unet()
    callbacks = Callbacks()
    train = pl.Trainer(max_epochs=200, accelerator="gpu", devices=1, callbacks=[callbacks])
    train.fit(model, mri_data)
    train.test(model, mri_data)

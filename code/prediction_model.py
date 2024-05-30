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
        pass # spit out some kind of csv file where data is represented as {subject id, predicted age, actual age, difference}

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
        self.o_1 = Network_Utility.convolution(1, 16)
        self.o_2 = Network_Utility.convolution(16, 32)
        self.o_3 = Network_Utility.convolution(32, 64)
        self.o_4 = Network_Utility.convolution(64, 128)
        self.o_5 = Network_Utility.convolution(128, 256)
        self.o_6 = Network_Utility.fcn_layers()
    
    # forward pass
    def forward(self, image):
        # naming convention: x_number of channels_encode/decode_up/down/nothing(side convolution)
        
        conv_1 = self.o_1(image)
        conv_2 = self.o_2(conv_1)
        conv_3 = self.o_3(conv_2)
        conv_4 = self.o_4(conv_3)
        conv_5 = self.o_5(conv_4)
        age_prediction = self.o_6(conv_5)
        return age_prediction

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

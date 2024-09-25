import os
from torch import optim, nn
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
from network_utility import network_utility
from hybrid3_utility import StandardFusion

# important folders
x_directory = "/home/ramkumars@acct.upmchs.net/Projects/Harmonizing-MRI-Scans/data/processed_input/" # CHANGE FOR WHEN USING JENKINS
# y_file = insert whatever json/text file has subject ID and age


class Callbacks():
    def on_test_end(self, trainer, pl_module):
        pass # spit out some kind of csv file where data is represented as {subject id, predicted age, actual age, difference}

class Fusion():
    def __init__(self, fusion_block: StandardFusion, learning_rate: int = 1e-3, criterion = nn.MSELoss):
        super().__init__()
        # hyperparameters
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.testing_outputs = []
        self.validation_outputs = []
        self.layers = []
        self.run = None
        self.fusion_block = fusion_block
    
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
    

class MRIDataModule():
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
        self.ground_truth_ages = []

    def setup(self, stage: str):
        # set up training, testing, validation split
        
        lens = network_utility.create_data_splits(len(self.input_files))
        training_stop_index = lens[0]
        testing_stop_index = lens[0] + lens[1]
        validation_stop_index = lens[0] + lens[1] + lens[2] - 1

        self.training = self.input_files[:training_stop_index]
        self.ground_truth_training = self.ground_truth_ages[:training_stop_index]

        self.training_dataset = MRIDataset(self.training, self.ground_truth_training)
        self.training_dataloader = self.train_dataloader()

        self.testing = self.input_files[training_stop_index:testing_stop_index]
        self.ground_truth_testing = self.ground_truth_ages[training_stop_index:testing_stop_index]  

        self.testing_dataset = MRIDataset(self.testing, self.ground_truth_testing)
        self.testing_dataloader = self.test_dataloader()

        self.validation = self.input_files[testing_stop_index:validation_stop_index] 
        self.ground_truth_validation = self.ground_truth_ages[testing_stop_index:validation_stop_index]

        self.validation_dataset = MRIDataset(self.validation, self.ground_truth_validation)
        self.validation_dataloader = self.val_dataloader()


    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size = self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.testing_dataset, batch_size = self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size = self.batch_size)

class MRIDataset(Dataset):
    def __init__(self, model_input: list = [], ground_truth: list = []):
        super().__init__()
        self.model_input = model_input
        self.ground_truth = ground_truth

    def __len__(self):
        return len(self.model_input)

    def __getitem__(self, index):
        scan_path = self.model_input[index]
        intermediate_scan_list = os.listdir(f"{x_directory}/{scan_path}/anat/")
        scan = nib.load(f"{x_directory}/{scan_path}/anat/{intermediate_scan_list[0]}")
        scan_array = scan.get_fdata()
        scan_tensor = torch.tensor(scan_array, dtype=torch.float32)
        slice_index = network_utility.get_slice(scan_tensor=scan_tensor)
        scan_slice = scan_tensor[:, :, slice_index]
        scan_slice = scan_slice[None, :, :]

        # ------------------------------------- Write code to process age here!!!!!--------------------------------------------
        age = None

        return {"scan": scan_slice}

if __name__ == "__main__":
    mri_data = MRIDataModule(batch_size=4)
    model = Fusion()
    callbacks = Callbacks()
    # train = pl.Trainer(max_epochs=200, accelerator="gpu", devices=1, callbacks=[callbacks])
    # train.fit(model, mri_data)
    # train.test(model, mri_data)



import copy
import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from cnn_utility import network_utility
from network_utility import Nu as nu
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt

class CNN(pl.LightningModule):
    def __init__(self, lr=1e-5):
        super().__init__()
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.testing_outputs = []
        self.validation_outputs = []

        # CNN architecture
        self.o_1 = network_utility.convolution(1, 16)
        self.o_2 = network_utility.convolution(16, 32)
        self.o_3 = network_utility.convolution(32, 64)
        self.o_4 = network_utility.convolution(64, 128)
        self.o_5 = network_utility.convolution(128, 256)
        self.o_6 = network_utility.fcn_layers()

    def forward(self, image):
        conv_1 = self.o_1(image)
        conv_2 = self.o_2(conv_1)
        conv_3 = self.o_3(conv_2)
        conv_4 = self.o_4(conv_3)
        conv_5 = self.o_5(conv_4)
        conv_5_flat = conv_5.view(conv_5.size(0), -1)
        age_prediction = self.o_6(conv_5_flat)
        return age_prediction

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x = train_batch["scan"]
        y = train_batch["age"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["scan"]
        y = val_batch["age"]
        y_hat = self(x)
        self.validation_outputs.append(y_hat)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        x = test_batch["scan"]
        y = test_batch["age"]
        y_hat = self(x)
        self.testing_outputs.append(y_hat)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)

class DTIDataset(Dataset):
    def __init__(self, model_input: list = [], brain_ages: list = [], dti_directory: str = ""):
        super().__init__()
        self.model_input = model_input
        self.ages = brain_ages
        self.dti_directory = dti_directory

    def __len__(self):
        return len(self.model_input)

    def __getitem__(self, index):
        scan_path = self.model_input[index]
        scan = nib.load(f"{self.dti_directory}/{scan_path}")
        scan_array = scan.get_fdata()
        scan_tensor = torch.tensor(scan_array, dtype=torch.float32)
        slice_index = nu.get_slice(scan_tensor=scan_tensor)
        scan_slice = scan_tensor[:, :, slice_index]
        scan_slice = scan_slice[None, :, :]

        age = self.ages[index]

        return {"scan": scan_slice, "age": age}

class DTIDataModule(pl.LightningDataModule):
    def __init__(self, dti_paths, ages_directory, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.dti_directory = dti_paths
        self.ages_directory = ages_directory
        self.input_files = os.listdir(dti_paths)
        self.ages = nu.import_data(ages_directory, self.input_files)
        self.training_dataloader = None
        self.testing_dataloader = None
        self.validation_dataloader = None
        self.training_dataset = None
        self.testing_dataset = None
        self.validation_dataset = None

    def setup(self, stage: str):
        # set up training, testing, validation split
        lens = nu.create_data_splits(len(self.input_files))
        training_stop_index = lens[0]
        testing_stop_index = lens[0] + lens[1]
        validation_stop_index = lens[0] + lens[1] + lens[2]

        training = self.input_files[:training_stop_index]
        training_ages = self.ages[:training_stop_index]

        self.training_dataset = DTIDataset(training, training_ages, self.dti_directory)
        self.training_dataloader = self.train_dataloader()

        testing = self.input_files[training_stop_index:testing_stop_index]
        testing_ages = self.ages[training_stop_index:testing_stop_index]

        self.testing_dataset = DTIDataset(testing, testing_ages, self.dti_directory)
        self.testing_dataloader = self.test_dataloader()

        validation = self.input_files[testing_stop_index:validation_stop_index]
        validation_ages = self.ages[testing_stop_index:validation_stop_index]

        self.validation_dataset = DTIDataset(validation, validation_ages, self.dti_directory)
        self.validation_dataloader = self.val_dataloader()

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.testing_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)

class SaveTensorBoardCallback(Callback):
    def __init__(self, log_dir, export_dir):
        self.log_dir = log_dir
        self.export_dir = export_dir

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        os.makedirs(self.export_dir, exist_ok=True)
        ea = EventAccumulator(self.log_dir)
        ea.Reload()

        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            df = pd.DataFrame(events)

            # Save to CSV
            csv_path = os.path.join(self.export_dir, f"{tag}.csv")
            df.to_csv(csv_path, index=False)

            # Create and save plot
            plt.figure(figsize=(10, 6))
            plt.plot(df['step'], df['value'])
            plt.title(tag)
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.savefig(os.path.join(self.export_dir, f"{tag}.png"))
            plt.close()

if __name__ == "__main__":
    print("initializing hyperparameters\n")
    # Hyperparameters
    learning_rate = 1e-5
    batch_size = 4
    max_epochs = 100

    # Directories
    dti_directory = "/ix1/haizenstein/shr120/data/data/CamCAN/"
    ages_directory = "/ix1/haizenstein/data/data/participant_data.csv"

    print("initializing tb and logger\n")
    # Logger and callbacks
    logger = TensorBoardLogger(save_dir="/ihome/haizenstein/shr120/lib/logs/", name="logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath="/ihome/haizenstein/shr120/lib/checkpoints/",
        filename="model-{epoch:02d}",
        save_top_k=1,
        every_n_epochs=10
    )
    tb_callback = SaveTensorBoardCallback(
        log_dir=logger.log_dir,
        export_dir="/ihome/haizenstein/shr120/lib/logs/cnn/version_1/"
    )

    print("initializing model\n")
    # Initialize model
    model = CNN(lr=learning_rate)

    print("initializing trainer\n")
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, tb_callback],
        logger=logger
    )

    # Initialize data module
    data_module = DTIDataModule(batch_size=batch_size, dti_paths=dti_directory, ages_directory=ages_directory)

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)
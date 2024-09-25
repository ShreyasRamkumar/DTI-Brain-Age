import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from network_utility import network_utility as nu
from cnn_prediction import CNN
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

dti_directory = "ix1/haizenstein/shr120/data/data/CamCAN"
ages_directory = "ix1/haizenstein/data/data/participant_data.csv"
arch = CNN()

class SaveTensorBoardCallback(Callback):
    def __init__(self, log_dir, export_dir):
        self.log_dir = log_dir
        self.export_dir = export_dir

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        os.makedirs(pl_module.log_dir, exist_ok=True)
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


class AgePredictor(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, model_arch=None):
        super().__init__()
        self.lr = learning_rate
        self.criterion = nn.MSELoss()
        self.testing_outputs = []
        self.validation_outputs = []
        self.arch = model_arch

    def forward(self, x):
        return self.arch(x)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        return opt

    def training_step(self, train_batch, batch_idx):
        x = train_batch["scan"]
        y = train_batch["age"]
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


class DTIDataModule(pl.LightningDataModule):
    def __init__(self, dti_paths, batch_size: int = 4):
        super().__init__()
        self.batch_size = batch_size
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
        validation_stop_index = lens[0] + lens[1] + lens[2] - 1

        training = self.input_files[:training_stop_index]
        training_ages = self.ages[:training_stop_index]

        self.training_dataset = DTIDataset(training, training_ages)
        self.training_dataloader = self.train_dataloader()

        testing = self.input_files[training_stop_index:testing_stop_index]
        testing_ages = self.ages[training_stop_index:testing_stop_index]

        self.testing_dataset = DTIDataset(testing, testing_ages)
        self.testing_dataloader = self.test_dataloader()

        validation = self.input_files[testing_stop_index:validation_stop_index]
        validation_ages = self.ages[testing_stop_index:validation_stop_index]

        self.validation_dataset = DTIDataset(validation, validation_ages)
        self.validation_dataloader = self.val_dataloader()

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.testing_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)


class DTIDataset(Dataset):
    def __init__(self, model_input: list = [], brain_ages: list = []):
        super().__init__()
        self.model_input = model_input
        self.ages = brain_ages

    def __len__(self):
        return len(self.model_input)

    def __getitem__(self, index):
        scan_path = self.model_input[index]
        scan = nib.load(f"{ages_directory}/{scan_path}")
        scan_array = scan.get_fdata()
        scan_tensor = torch.tensor(scan_array, dtype=torch.float32)
        slice_index = nu.get_slice(scan_tensor=scan_tensor)
        scan_slice = scan_tensor[:, :, slice_index]
        scan_slice = scan_slice[None, :, :]

        age = self.ages[index]

        return {"scan": scan_slice, "age": age}


if __name__ == "__main__":
    logger = TensorBoardLogger(save_dir="ihome/haizenstein/shr120/lib/logs/", name="logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath="ihome/haizenstein/shr120/lib/checkpoints/",
        filename="model-{epoch:02d}",
        save_top_k=1,
        every_n_epochs=10
    )
    tb_callback = SaveTensorBoardCallback(
        log_dir=logger.log_dir,
        export_dir="ihome/haizenstein/shr120/lib/logs/cnn/version_1/"
    )

    dti_data = DTIDataModule(batch_size=4, dti_paths=dti_directory)
    model = AgePredictor(model_arch=arch)
    train = pl.Trainer(max_epochs=200, accelerator="gpu", devices=1, callbacks=[checkpoint_callback, tb_callback])
    train.fit(model, dti_data)
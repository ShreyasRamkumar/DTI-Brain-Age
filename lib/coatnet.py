import torch
from torch import optim
from coatnet_utility import *
import pytorch_lightning as pl
import nibabel as nib
from network_utility import Nu as nu
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch.utils.data import DataLoader, Dataset
import os
from pytorch_lightning.loggers import TensorBoardLogger

# CoAtNet Architecture
class CoAtNet(pl.LightningModule):
    def __init__(self, lr=1e-5, num_classes=1):
        super(CoAtNet, self).__init__()
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.training_output = []
        self.testing_outputs = []
        self.validation_outputs = []

        # Stage S0: Stem
        self.stem = nn.Sequential(
            ConvBlock(1, 64, kernel_size=3, stride=2),  # 128x128 -> 64x64
            ConvBlock(64, 64, kernel_size=3, stride=1)
        )

        # Stage S1: Conv blocks
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),  # 64x64 -> 32x32
            ResidualBlock(128, 128)
        )

        # Stage S2: Conv blocks
        self.stage2 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),  # 32x32 -> 16x16
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )

        # Stage S3: Attention blocks
        self.stage3 = nn.Sequential(
            AttentionBlock(dim=256, heads=8),  # 16x16 -> 8x8
            AttentionBlock(dim=256, heads=8)
        )

        # Stage S4: Attention blocks
        self.stage4 = nn.Sequential(
            AttentionBlock(dim=256, heads=8),  # 8x8 -> 4x4
            AttentionBlock(dim=256, heads=8)
        )

        # Global Pooling and Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Forward pass through each stage
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x = train_batch["scan"].to(self.device)
        y = train_batch["age"].float().to(self.device)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["scan"].to(self.device)
        y = val_batch["age"].float().to(self.device)
        y_hat = self(x)
        self.validation_outputs.append(y_hat)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x = test_batch["scan"].to(self.device)
        y = test_batch["age"].float().to(self.device)
        y_hat = self(x)
        self.testing_outputs.append(y_hat)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)
        return loss

class DTIDataset(Dataset):
    def __init__(self, model_input, brain_ages, dti_directory):
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
    def __init__(self, dti_directory, ages_csv, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.dti_directory = dti_directory

        print("importing data from scans and ages")
        paths = os.listdir(dti_directory)
        self.data = nu.import_data(ages_csv, SortedDict.fromkeys(paths))

        self.training_dataloader = None
        self.testing_dataloader = None
        self.validation_dataloader = None
        self.training_dataset = None
        self.testing_dataset = None
        self.validation_dataset = None

    def setup(self, stage: str):
        # set up training, testing, validation split
        lens = nu.create_data_splits(len(self.data))
        training_stop_index = lens[0]
        testing_stop_index = lens[0] + lens[1]
        validation_stop_index = lens[0] + lens[1] + lens[2]

        training = dict(list(self.data.items())[:training_stop_index])

        self.training_dataset = DTIDataset(list(training.keys()), list(training.values()), self.dti_directory)
        self.training_dataloader = self.train_dataloader()

        testing = dict(list(self.data.items())[training_stop_index:testing_stop_index])

        self.testing_dataset = DTIDataset(list(testing.keys()), list(testing.values()), self.dti_directory)
        self.testing_dataloader = self.test_dataloader()

        validation = dict(list(self.data.items())[testing_stop_index:validation_stop_index])

        self.validation_dataset = DTIDataset(list(validation.keys()), list(validation.values()), self.dti_directory)
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
    dti_directory = "C:\Code\GPN\DTI-Brain-Age\data\CamCAN" # "/ix1/haizenstein/shr120/data/CamCAN/"
    ages_csv = "C:\Code\GPN\DTI-Brain-Age\data\participant_data.csv" # "/ix1/haizenstein/shr120/data/participant_data.csv"

    print("initializing tb and logger\n")
    # Logger and callbacks
    # logger = TensorBoardLogger(save_dir="/ihome/haizenstein/shr120/lib/logs/", name="logs")
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="/ihome/haizenstein/shr120/lib/checkpoints/",
    #     filename="model-{epoch:02d}",
    #     save_top_k=1,
    #     every_n_epochs=10
    # )
    # tb_callback = SaveTensorBoardCallback(
    #     log_dir=logger.log_dir,
    #     export_dir="/ihome/haizenstein/shr120/lib/logs/cnn/version_1_extended/"
    # )

    print("initializing model\n")
    # Initialize model
    model = CoAtNet()

    print("initializing trainer\n")
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        # callbacks=[checkpoint_callback, tb_callback],
        # logger=logger
    )

    # Initialize data module
    print("initializing data module\n")
    data_module = DTIDataModule(batch_size=batch_size, dti_directory=dti_directory, ages_csv=ages_csv)

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

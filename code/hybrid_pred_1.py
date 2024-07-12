import os
import lightning.pytorch as pl
from torch import optim, nn
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
from lightning.pytorch.callbacks import Callback
from network_utility import network_utility


# important folders
x_directory = "/home/ramkumars@acct.upmchs.net/Projects/Harmonizing-MRI-Scans/data/processed_input/" # CHANGE FOR WHEN USING JENKINS
# y_file = insert whatever json/text file has subject ID and age


class Callbacks(Callback):
    def on_test_end(self, trainer, pl_module):
        pass # spit out some kind of csv file where data is represented as {subject id, predicted age, actual age, difference}
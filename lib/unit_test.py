from cnn_prediction import CNN
from network_utility import network_utility
from hybrid_pred_1 import CoAtNet
import nibabel as nib
from torch import from_numpy, unsqueeze, randn
from Preprocessing import Preprocessing


# cnn = CNN() ###### WORKS ########
# ###### WORKS #######
# ctnet = CoAtNet()

scan = nib.load("C:\\Code\\GPN\\DTI-Brain-Age\\testing\\fa_testing.nii")
scan_array = scan.get_fdata()
scan_tensor = from_numpy(scan_array)
slice_int = network_utility.get_slice(scan_tensor)

slice_tensor = scan_tensor[:, :, slice_int].float()
slice_tensor = unsqueeze(slice_tensor, 0)
slice_tensor = unsqueeze(slice_tensor, 0)

padded_tensor = Preprocessing.pad_to_128(img_tensor=slice_tensor)

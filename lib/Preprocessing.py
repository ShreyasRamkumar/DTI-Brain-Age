from dipy.segment.threshold import otsu
import nibabel as nib
from numpy import where, eye, multiply
from nilearn.masking import compute_brain_mask
from torch import from_numpy
from torch.nn.functional import pad



class Preprocessing:
    def __init__(self, image_path: str):
        self.image_path = image_path

    def isolate_white_matter(self):
        image = nib.load(self.image_path)
        image_array = image.get_fdata()

        iso_mask = compute_brain_mask(target_img=image, threshold=0.1, mask_type="wm")
        mask_array = iso_mask.get_fdata()
        
        iso_array = multiply(image_array, mask_array)
        
        return from_numpy(iso_array).float()

    @staticmethod
    def pad_to_128(img_tensor):
        """
        Pads the input tensor to height and width of 96, while keeping the depth unchanged.

        Parameters:
        img_tensor (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
        torch.Tensor: Padded tensor of shape (N, C, 96, 96)
        """
        N, C, H, W = img_tensor.shape

        pad_h = 128 - H
        pad_w = 128 - W

        # Check if padding is needed
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Height or Width of the input tensor is greater than 96.")

        # Calculate the padding values for height and width to pad equally on both sides
        pad_h_top = pad_h // 2
        pad_h_bottom = pad_h - pad_h_top
        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        # Apply zero padding
        img_tensor_padded = pad(img_tensor,
                                (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom),
                                mode='constant',
                                value=0)

        return img_tensor_padded

if __name__ == "__main__":
    p = Preprocessing("C:/Code/GPN/DTI-Brain-Age/testing/fa_testing.nii")
    p.isolate_white_matter()
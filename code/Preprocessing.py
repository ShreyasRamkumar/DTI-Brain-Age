from dipy.segment.threshold import otsu
import nibabel as nib
from numpy import where, eye, multiply
from nilearn.masking import compute_brain_mask



class Preprocessing:
    def __init__(self, image_path: str):
        self.image_path = image_path

    def isolate_white_matter(self):
        image = nib.load(self.image_path)
        image_array = image.get_fdata()

        iso_mask = compute_brain_mask(target_img=image, threshold=0.1, mask_type="wm")
        mask_array = iso_mask.get_fdata()
        
        iso_array = multiply(image_array, mask_array)
        
        iso_image = nib.Nifti1Image(iso_array, eye(4))
        nib.save(iso_image, "C:/Code/GPN/DTI-Brain-Age/testing/nilearn_fa_testing.nii")

if __name__ == "__main__":
    p = Preprocessing("C:/Code/GPN/DTI-Brain-Age/testing/fa_testing.nii")
    p.isolate_white_matter()
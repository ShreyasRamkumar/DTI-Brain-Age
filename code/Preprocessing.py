from dipy.segment.threshold import otsu
import nibabel as nib
from numpy import where, eye

class Preprocessing:
    def __init__(self, image_path: str):
        self.image_path = image_path

    def isolate_white_matter(self):
        # get otsu value from FA map
        image = nib.load(self.image_path)
        image_array = image.get_fdata()
        otsu_val = otsu(image_array)
        
        # use otsu value to isolate white matter
        iso_array = where(image_array >= otsu_val, image_array, 0)
        iso_image = nib.Nifti1Image(iso_array, eye(4))
        nib.save(iso_image, "C:/Code/GPN/DTI-Brain-Age/testing/isolated_fa_testing.nii")

if __name__ == "__main__":
    p = Preprocessing("C:/Code/GPN/DTI-Brain-Age/testing/fa_testing.nii")
    p.isolate_white_matter()
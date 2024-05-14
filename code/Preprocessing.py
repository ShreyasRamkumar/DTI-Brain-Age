from dipy.segment.threshold import otsu
import nibabel as nib

class Preprocessing:
    def __init__(self, image_path: str):
        self.image_path = image_path

    def isolate_white_matter(self):
        image = nib.load(self.image_path)
        image_array = image.get_fdata()
        otsu_val = otsu(image_array)
        
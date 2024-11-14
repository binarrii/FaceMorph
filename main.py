import os

from contrib.Comfyui_face_parsing.face_parsing_nodes import BBoxDetect, BBoxDetectorLoader, BBoxListItemSelect, ImageCropWithBBox, ImageInsertWithBBox
from contrib.ComfyUI_fnodes.Face_morph import FaceMorph
from contrib.ComfyUI_fnodes.utils.image_convert import pil2tensor
from PIL import Image

source_image_path = "/mnt/onas/share4/frame_after_swap_for_post_processing/ep-测试/female"
target_image_path = "/mnt/onas/share4/frame_after_swap_for_post_processing/ep-测试/female_predict_face"

if __name__ == "__main__":
    for root, dirs, files in os.walk(source_image_path):
        for f in files:
            source_image = pil2tensor(Image.open(f"{source_image_path}/{f}"))
            target_image = pil2tensor(Image.open(f"{target_image_path}/{f}"))

            face_morph = FaceMorph()
            warped_image = face_morph.execute(source_image, target_image,
                landmark_type="OUTLINE", align_type="Landmarks", onnx_device="CUDA")
            

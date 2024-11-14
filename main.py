import os
import traceback

from contrib.ComfyUI_face_parsing.face_parsing_nodes import BBoxDetect, ImageCropWithBBox, ImageInsertWithBBox
from contrib.ComfyUI_fnodes.face_morph import FaceMorph
from contrib.ComfyUI_fnodes.utils.image_convert import pil2tensor, tensor2pil
from PIL import Image

_SOURCE_IMAGE_PATH = "/mnt/onas/share4/frame_after_swap_for_post_processing/ep-测试/female"
_TARGET_IMAGE_PATH = "/mnt/onas/share4/frame_after_swap_for_post_processing/ep-测试/female_predict_face"
_RESULT_IMAGE_PATH = "/mnt/onas/share4/frame_after_swap_for_post_processing/ep-测试/female_morphed"

base_path = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(base_path, "models")

if __name__ == "__main__":
    from ultralytics import YOLO
    bbox_detector = YOLO(model=os.path.join(models_dir, "ultralytics/bbox/face_yolov8m.pt"))

    bbox_detect = BBoxDetect()
    image_crop_with_bbox = ImageCropWithBBox()
    image_insert_with_bbox = ImageInsertWithBBox()
    face_morph = FaceMorph()

    for root, dirs, files in os.walk(_SOURCE_IMAGE_PATH):
        for f in filter(lambda x: x.endswith('.png'), files):
            source_image_tensor = pil2tensor(Image.open(f"{_SOURCE_IMAGE_PATH}/{f}"))
            target_image_tensor = pil2tensor(Image.open(f"{_TARGET_IMAGE_PATH}/{f}"))

            # noinspection PyBroadException
            try:
                source_bboxes, _ = bbox_detect.main(
                    bbox_detector=bbox_detector,
                    image=source_image_tensor,
                    threshold=0.3,
                    dilation=20,
                    dilation_ratio=0.3,
                    by_ratio=True,
                )
                cropped_source_image_tensors, = image_crop_with_bbox.main(
                    bbox=source_bboxes[0],
                    image=source_image_tensor
                )

                target_bboxes, _ = bbox_detect.main(
                    bbox_detector=bbox_detector,
                    image=target_image_tensor,
                    threshold=0.3,
                    dilation=20,
                    dilation_ratio=0.3,
                    by_ratio=True,
                )
                cropped_target_image_tensors, = image_crop_with_bbox.main(
                    bbox=target_bboxes[0],
                    image=target_image_tensor
                )

                warped_image_tensor, = face_morph.execute(
                    source_image=cropped_source_image_tensors,
                    target_image=cropped_target_image_tensors,
                    landmark_type="OUTLINE",
                    align_type="Landmarks",
                    onnx_device="CUDA",
                )
                final_image_tensor, = image_insert_with_bbox.main(
                    bbox=target_bboxes[0],
                    image_src=source_image_tensor,
                    image=warped_image_tensor
                )
                tensor2pil(final_image_tensor).save(f"{_RESULT_IMAGE_PATH}/{f}")

                print(f"\033[92m[FaceMorph] source: {_SOURCE_IMAGE_PATH}/{f}")
                print(f"\033[92m[FaceMorph] target: {_TARGET_IMAGE_PATH}/{f}")
                print(f"\033[92m[FaceMorph] result: {_RESULT_IMAGE_PATH}/{f}")
            except:
                traceback.print_stack()

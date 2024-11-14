import os
import traceback

from PIL import Image
from contrib.ComfyUI_face_parsing.face_parsing_nodes import BBoxDetect, ImageCropWithBBox, ImageInsertWithBBox
from contrib.ComfyUI_fnodes.face_morph import FaceMorph
from contrib.ComfyUI_fnodes.utils.image_convert import pil2tensor, tensor2pil, pil2np
from deepface import DeepFace

_CGREEN, _CRED, _CEND = "\033[92m", "\033[91m", "\033[97m"

_SOURCE_IMAGE_PATH = "/mnt/onas/share4/frame_after_swap_for_post_processing/ep-测试/female"
_TARGET_IMAGE_PATH = "/mnt/onas/share4/frame_after_swap_for_post_processing/ep-测试/female_predict_face"
_RESULT_IMAGE_PATH = "/mnt/onas/share4/frame_after_swap_for_post_processing/ep-测试/female_morphed"

_REF_FACE_PATH = "/mnt/onas/share4/frame_after_swap_for_post_processing/ep-测试/female.png"

base_path = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(base_path, "models")

if __name__ == "__main__":
    from ultralytics import YOLO

    bbox_detector = YOLO(model=os.path.join(models_dir, "ultralytics/bbox/face_yolov8m.pt"))

    bbox_detect = BBoxDetect()
    image_crop_with_bbox = ImageCropWithBBox()
    image_insert_with_bbox = ImageInsertWithBBox()
    face_morph = FaceMorph()

    ref_face_np = pil2np(Image.open(_REF_FACE_PATH).convert('RGB'))


    def png_or_jpg(filename):
        return filename.endswith('.png') or filename.endswith('.jpg')


    for root, dirs, files in os.walk(_SOURCE_IMAGE_PATH):
        for f in filter(png_or_jpg, files):
            source_image_tensor = pil2tensor(Image.open(f"{_SOURCE_IMAGE_PATH}/{f}"))
            target_image_tensor = pil2tensor(Image.open(f"{_TARGET_IMAGE_PATH}/{f}"))

            # noinspection PyBroadException
            try:
                source_bboxes, source_cnt = bbox_detect.main(
                    bbox_detector=bbox_detector,
                    image=source_image_tensor,
                    threshold=0.5,
                    dilation=20,
                    dilation_ratio=0.3,
                    by_ratio=True,
                )
                if source_cnt <= 0:
                    print(f"{_CRED}[FaceMorph] no face detected in source: {_SOURCE_IMAGE_PATH}/{f}{_CEND}")
                    continue

                target_bboxes, target_cnt = bbox_detect.main(
                    bbox_detector=bbox_detector,
                    image=target_image_tensor,
                    threshold=0.5,
                    dilation=20,
                    dilation_ratio=0.3,
                    by_ratio=True,
                )
                if target_cnt <= 0:
                    print(f"{_CRED}[FaceMorph] no face detected in target: {_TARGET_IMAGE_PATH}/{f}{_CEND}")
                    continue

                cropped_source_image_tensor, cropped_target_image_tensor = None, None
                for source_bbox in source_bboxes:
                    cropped_source_image_tensor, = image_crop_with_bbox.main(
                        bbox=source_bbox,
                        image=source_image_tensor
                    )
                    same_face = DeepFace.verify(
                        ref_face_np,
                        pil2np(tensor2pil(cropped_source_image_tensor)),
                        model_name="Facenet512",
                        detector_backend="yolov8",
                        enforce_detection=False
                    )
                    if same_face["verified"]:
                        break
                if cropped_source_image_tensor is None:
                    print(f"{_CRED}[FaceMorph] no face matched in source: {_SOURCE_IMAGE_PATH}/{f}{_CEND}")
                    continue

                for target_bbox in target_bboxes:
                    cropped_target_image_tensor, = image_crop_with_bbox.main(
                        bbox=target_bbox,
                        image=target_image_tensor
                    )
                    same_face = DeepFace.verify(
                        ref_face_np,
                        pil2np(tensor2pil(cropped_target_image_tensor)),
                        model_name="Facenet512",
                        detector_backend="yolov8",
                        enforce_detection=False
                    )
                    if same_face["verified"]:
                        break
                if cropped_target_image_tensor is None:
                    print(f"{_CRED}[FaceMorph] no face matched in target: {_TARGET_IMAGE_PATH}/{f}{_CEND}")
                    continue

                warped_image_tensor, = face_morph.execute(
                    source_image=cropped_source_image_tensor,
                    target_image=cropped_target_image_tensor,
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

                print(f"{_CGREEN}[FaceMorph] source: {_SOURCE_IMAGE_PATH}/{f}{_CEND}")
                print(f"{_CGREEN}[FaceMorph] target: {_TARGET_IMAGE_PATH}/{f}{_CEND}")
                print(f"{_CGREEN}[FaceMorph] result: {_RESULT_IMAGE_PATH}/{f}{_CEND}")
            except:
                traceback.print_exc()

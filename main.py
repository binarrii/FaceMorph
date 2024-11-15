import argparse
import os
import time
import traceback

from PIL import Image
from contrib.ComfyUI_face_parsing.face_parsing_nodes import BBoxDetect, ImageCropWithBBox, ImageInsertWithBBox
from contrib.ComfyUI_fnodes.face_morph import FaceMorph
from contrib.ComfyUI_fnodes.utils.image_convert import pil2tensor, tensor2pil, pil2np
from deepface import DeepFace


_CGREEN, _CRED, _CYELLOW, _CEND = "\033[92m", "\033[91m", "\033[93m", "\033[97m"

base_path = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(base_path, "models")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Morph')
    parser.add_argument('--workdir', required=True, type=str, default="")
    parser.add_argument('--refface', required=True, type=str, default="")
    args, _ = parser.parse_known_args()

    _SOURCE_IMAGE_PATH = f"{args.workdir}/{args.refface}"
    _TARGET_IMAGE_PATH = f"{args.workdir}/{args.refface}_predict_face"
    _RESULT_IMAGE_PATH = f"{args.workdir}/{args.refface}_morphed_face"

    if not os.path.exists(_RESULT_IMAGE_PATH):
        os.makedirs(_RESULT_IMAGE_PATH, exist_ok=True)

    _REF_FACE_PATH = f"{args.workdir}/{args.refface}.png"
    if not os.path.exists(_REF_FACE_PATH):
        _REF_FACE_PATH = f"{args.workdir}/{args.refface}.jpg"

    from ultralytics import YOLO
    bbox_detector = YOLO(model=os.path.join(models_dir, "ultralytics/bbox/face_yolov8m.pt"))

    bbox_detect = BBoxDetect()
    image_crop_with_bbox = ImageCropWithBBox()
    image_insert_with_bbox = ImageInsertWithBBox()
    face_morph = FaceMorph()

    ref_face_np = pil2np(Image.open(_REF_FACE_PATH).convert('RGB'))


    def png_or_jpg(filename: str):
        return (not filename.startswith('.')) and \
            (filename.endswith('.png') or filename.endswith('.jpg'))


    for root, dirs, files in os.walk(_SOURCE_IMAGE_PATH):
        for f in filter(png_or_jpg, files):
            t0 = time.time()

            source_image_tensor = pil2tensor(Image.open(f"{_SOURCE_IMAGE_PATH}/{f}"))
            target_image_tensor = pil2tensor(Image.open(f"{_TARGET_IMAGE_PATH}/{f}"))

            # noinspection PyBroadException
            try:
                source_bboxes, source_cnt = bbox_detect.main(
                    bbox_detector=bbox_detector,
                    image=source_image_tensor,
                    threshold=0.5,
                    dilation=20,
                    dilation_ratio=0.5,
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
                    dilation_ratio=0.5,
                    by_ratio=True,
                )
                if target_cnt <= 0:
                    print(f"{_CRED}[FaceMorph] no face detected in target: {_TARGET_IMAGE_PATH}/{f}{_CEND}")
                    continue

                cropped_source_image_tensor, source_face_matched = None, False
                for source_bbox in source_bboxes:
                    cropped_source_image_tensor, = image_crop_with_bbox.main(
                        bbox=source_bbox,
                        image=source_image_tensor
                    )
                    same_face = DeepFace.verify(
                        ref_face_np,
                        pil2np(tensor2pil(cropped_source_image_tensor)),
                        model_name="VGG-Face",
                        detector_backend="skip",
                        enforce_detection=False
                    )
                    if same_face["verified"]:
                        source_face_matched = True
                        break
                if not source_face_matched:
                    print(f"{_CRED}[FaceMorph] no face matched in source: {_SOURCE_IMAGE_PATH}/{f}{_CEND}")
                    continue

                cropped_target_image_tensor, target_face_matched = None, False
                for target_bbox in target_bboxes:
                    cropped_target_image_tensor, = image_crop_with_bbox.main(
                        bbox=target_bbox,
                        image=target_image_tensor
                    )
                    same_face = DeepFace.verify(
                        ref_face_np,
                        pil2np(tensor2pil(cropped_target_image_tensor)),
                        model_name="VGG-Face",
                        detector_backend="skip",
                        enforce_detection=False
                    )
                    if same_face["verified"]:
                        target_face_matched = True
                        break
                if not target_face_matched:
                    print(f"{_CRED}[FaceMorph] no face matched in target: {_TARGET_IMAGE_PATH}/{f}{_CEND}")
                    continue

                warped_image_tensor, = face_morph.execute(
                    source_image=cropped_source_image_tensor,
                    target_image=cropped_target_image_tensor,
                    landmark_type="OUTLINE",
                    align_type="Landmarks",
                    onnx_device="torch_gpu",
                )

                final_image_tensor, = image_insert_with_bbox.main(
                    bbox=target_bbox,
                    image_src=source_image_tensor,
                    image=warped_image_tensor
                )
                tensor2pil(final_image_tensor).save(f"{_RESULT_IMAGE_PATH}/{f}")

                print(f"{_CGREEN}[FaceMorph] source: {_SOURCE_IMAGE_PATH}/{f}{_CEND}")
                print(f"{_CGREEN}[FaceMorph] target: {_TARGET_IMAGE_PATH}/{f}{_CEND}")
                print(f"{_CGREEN}[FaceMorph] result: {_RESULT_IMAGE_PATH}/{f}{_CEND}")
                print(f"{_CYELLOW}[FaceMorph] time taken: {f} {(time.time() - t0) * 1000}ms{_CEND}")
            except:
                traceback.print_exc()
                print(f"{_CRED}[FaceMorph] error occurred: {_SOURCE_IMAGE_PATH}/{f}{_CEND}")
                print(f"{_CRED}[FaceMorph] error occurred: {_TARGET_IMAGE_PATH}/{f}{_CEND}")

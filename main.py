import argparse
import glob
import multiprocessing
import os
import sys
import time
import traceback

from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor

from contrib.ComfyUI_face_parsing.face_parsing_nodes import BBoxDetect, ImageCropWithBBox, ImageInsertWithBBox
from contrib.ComfyUI_fnodes.face_morph import FaceMorph
from contrib.ComfyUI_fnodes.utils.image_convert import pil2tensor, tensor2pil, pil2np
from deepface import DeepFace
from PIL import Image
from torch import Tensor
from ultralytics import YOLO


_CGREEN, _CRED, _CYELLOW, _CEND = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp']
_YOLO_MODEL_PATH = "models/ultralytics/bbox/face_yolov8m.pt"


class FaceMorpher:

    def __init__(self, workdir: str, ref_face: str):
        self.ref_faces_np = self._ref_faces_np(workdir, ref_face)
        self.bbox_detector = YOLO(model=_YOLO_MODEL_PATH)
        self.bbox_detect = BBoxDetect()
        self.image_crop_with_bbox = ImageCropWithBBox()
        self.image_insert_with_bbox = ImageInsertWithBBox()
        self.face_morph = FaceMorph()

    @staticmethod
    def _ref_faces_np(workdir: str, ref_face: str):
        faces_np = []
        for ext in _IMAGE_EXTENSIONS:
            for f in glob.iglob(f"{ref_face}*{ext}", root_dir=workdir):
                face_pil = Image.open(f"{workdir}/{f}").convert('RGB')
                faces_np.append(pil2np(face_pil))
        if len(faces_np) > 0:
            return faces_np
        raise FileNotFoundError(f"`{ref_face}` not found in {workdir}")
    
    def _verify(self, image: Tensor):
        for ref_face_np in self.ref_faces_np:
            same_face = DeepFace.verify(
                ref_face_np,
                pil2np(tensor2pil(image)),
                model_name="VGG-Face",
                detector_backend="skip",
                enforce_detection=False
            )
            if same_face["verified"]:
                return True
        return False

    def __call__(self, source_image_path: str, target_image_path: str, result_image_path: str):
        """
        Returns: tuple[bool, str, str, str]
        """
        t0 = time.time()

        source_image_tensor = pil2tensor(Image.open(source_image_path))
        target_image_tensor = pil2tensor(Image.open(target_image_path))

        # noinspection PyBroadException
        try:
            source_bboxes, source_cnt = self.bbox_detect.main(
                bbox_detector=self.bbox_detector,
                image=source_image_tensor,
                threshold=0.5,
                dilation=20,
                dilation_ratio=0.3,
                by_ratio=True,
            )
            if source_cnt <= 0:
                tensor2pil(source_image_tensor).save(result_image_path)
                print(f"{_CRED}[FaceMorph] no face detected in source: {source_image_path}{_CEND}")
                return (False, source_image_path, target_image_path, None)

            target_bboxes, target_cnt = self.bbox_detect.main(
                bbox_detector=self.bbox_detector,
                image=target_image_tensor,
                threshold=0.5,
                dilation=20,
                dilation_ratio=0.3,
                by_ratio=True,
            )
            if target_cnt <= 0:
                tensor2pil(source_image_tensor).save(result_image_path)
                print(f"{_CRED}[FaceMorph] no face detected in target: {target_image_path}{_CEND}")
                return (False, source_image_path, target_image_path, None)

            cropped_source_image_tensor, source_face_matched = None, False
            for source_bbox in source_bboxes:
                cropped_source_image_tensor, = self.image_crop_with_bbox.main(
                    bbox=source_bbox,
                    image=source_image_tensor
                )
                if self._verify(cropped_source_image_tensor):
                    source_face_matched = True
                    break
            if not source_face_matched:
                tensor2pil(source_image_tensor).save(result_image_path)
                print(f"{_CRED}[FaceMorph] no face matched in source: {source_image_path}{_CEND}")
                return (False, source_image_path, target_image_path, None)

            cropped_target_image_tensor, target_face_matched = None, False
            for target_bbox in target_bboxes:
                cropped_target_image_tensor, = self.image_crop_with_bbox.main(
                    bbox=target_bbox,
                    image=target_image_tensor
                )
                if self._verify(cropped_target_image_tensor):
                    target_face_matched = True
                    break
            if not target_face_matched:
                tensor2pil(source_image_tensor).save(result_image_path)
                print(f"{_CRED}[FaceMorph] no face matched in target: {target_image_path}{_CEND}")
                return (False, source_image_path, target_image_path, None)

            warped_image_tensor, = self.face_morph.execute(
                source_image=cropped_source_image_tensor,
                target_image=cropped_target_image_tensor,
                landmark_type="OUTLINE",
                align_type="Landmarks",
                onnx_device="torch_gpu",
            )

            final_image_tensor, = self.image_insert_with_bbox.main(
                bbox=source_bbox,
                image_src=source_image_tensor,
                image=warped_image_tensor
            )
            tensor2pil(final_image_tensor).save(result_image_path)

            print(f"{_CGREEN}[FaceMorph] source: {source_image_path}{_CEND}")
            print(f"{_CGREEN}[FaceMorph] target: {target_image_path}{_CEND}")
            print(f"{_CGREEN}[FaceMorph] result: {result_image_path}{_CEND}")
            print(f"{_CYELLOW}[FaceMorph] time taken: \
                  {os.path.basename(source_image_path)} {(time.time() - t0) * 1000}ms{_CEND}")
        except:
            tensor2pil(source_image_tensor).save(result_image_path)
            traceback.print_exc()
            print(f"{_CRED}[FaceMorph] error occurred: {source_image_path}{_CEND}")
            print(f"{_CRED}[FaceMorph] error occurred: {target_image_path}{_CEND}")
            return (False, source_image_path, target_image_path, None)
        
        return (True, source_image_path, target_image_path, result_image_path)


_face_morpher = None

def _init_worker(args: Namespace):
    global _face_morpher
    try:
        if _face_morpher is None:
            _face_morpher = FaceMorpher(args.workdir, args.refface)
    except:
        traceback.print_exc()
        sys.exit(1)

def _morph_face(args: Namespace, f: str):
    source_image_path = f"{args.workdir}/{args.refface}/{f}"
    target_image_path = f"{args.workdir}/{args.refface}_predict_face/{f}"
    result_image_path = f"{args.workdir}/{args.refface}_morphed_face/{f}"
    return _face_morpher(source_image_path, target_image_path, result_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Morph')
    parser.add_argument('--workdir', required=True, type=str, default="")
    parser.add_argument('--refface', required=True, type=str, default="")
    parser.add_argument('--workers', required=False, type=int, default=2)
    args, _ = parser.parse_known_args()

    _SOURCE_IMAGE_PATH = f"{args.workdir}/{args.refface}"
    _RESULT_IMAGE_PATH = f"{args.workdir}/{args.refface}_morphed_face"
    _SKIPED_IMAGE_LIST = f"{args.workdir}/{args.refface}_skipped.txt"
    _N = min(8, max(1, args.workers))

    if not os.path.exists(_RESULT_IMAGE_PATH):
        os.makedirs(_RESULT_IMAGE_PATH, exist_ok=True)

    if os.path.exists(_SKIPED_IMAGE_LIST):
        os.remove(_SKIPED_IMAGE_LIST)

    def _png_or_jpg(filename: str):
        return (not filename.startswith('.')) and (os.path.splitext(filename)[1] in _IMAGE_EXTENSIONS)

    multiprocessing.set_start_method('spawn')
    with ProcessPoolExecutor(max_workers=_N, initializer=_init_worker, initargs=(args,)) as executor:
        future_results = []
        for roots, dirs, files in os.walk(_SOURCE_IMAGE_PATH):
            for f in filter(_png_or_jpg, files):
                future_results.append(executor.submit(_morph_face, args, f))
    
    with open(_SKIPED_IMAGE_LIST, 'a+') as skipped:
        for future in future_results:
            ok, source, target, result = future.result()
            if not ok:
                skipped.write(f"{source}\n")

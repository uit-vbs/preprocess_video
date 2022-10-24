import os
import cv2
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from modules import Preprocess, Detection, OCR, Correction
from tool.config import Config
from tool.utils import natural_keys
import time


def init_args():
    parser = argparse.ArgumentParser("Document Extraction")
    parser.add_argument("--output", default="./results", help="Path to output folder")
    parser.add_argument(
        "--debug", action="store_true", help="Save every steps for debugging"
    )
    parser.add_argument(
        "--find_best_rotation",
        action="store_true",
        help="Whether to find rotation of document in the image",
    )
    return parser.parse_args()


class Pipeline:
    def __init__(self, args, config, device, gpu_id):
        self.output = args.output
        self.debug = args.debug
        self.find_best_rotation = args.find_best_rotation
        self.det_config_dir = '/app/vietnamese-ocr-toolbox/tool/config/detection/mmocr_configs/'
        self.device = device
        self.gpu_id = gpu_id
        self.det_model_name = 'DRRG'
        self.load_config(config)
        self.make_cache_folder()
        self.init_modules()

    def load_config(self, config):
        self.det_weight = config.det_weight
        self.ocr_weight = config.ocr_weight
        self.det_config = config.det_config
        self.ocr_config = config.ocr_config
        self.bert_weight = config.bert_weight
        self.class_mapping = {k: v for v, k in enumerate(config.retr_classes)}
        self.idx_mapping = {v: k for k, v in self.class_mapping.items()}
        self.dictionary_path = config.dictionary_csv
        self.retr_mode = config.retr_mode
        self.correction_mode = config.correction_mode

    def make_cache_folder(self):
        self.cache_folder = os.path.join(self.output, f"cache_process_{self.gpu_id}")
        os.makedirs(self.cache_folder, exist_ok=True)
        self.preprocess_cache = os.path.join(self.cache_folder, "preprocessed.jpg")
        self.detection_cache = os.path.join(self.cache_folder, "detected.jpg")
        self.crop_cache = os.path.join(self.cache_folder, "crops")
        os.makedirs(self.crop_cache, exist_ok=True)
        self.final_output = os.path.join(self.output, "result.jpg")
        self.retr_output = os.path.join(self.output, "result.txt")

    def init_modules(self):
        self.det_model = Detection(det_model_name=self.det_model_name, config_dir=self.det_config_dir, device=self.device)
        self.ocr_model = OCR(config_path=self.ocr_config, weight_path=self.ocr_weight, device=self.device)
        self.preproc = Preprocess(
            det_model=self.det_model,
            ocr_model=self.ocr_model,
            find_best_rotation=self.find_best_rotation,
        )

        if self.dictionary_path is not None:
            self.dictionary = {}
            df = pd.read_csv(self.dictionary_path)
            for id, row in df.iterrows():
                self.dictionary[row.text.lower()] = row.lbl
        else:
            self.dictionary = None

        self.correction = Correction(
            dictionary=self.dictionary, mode=self.correction_mode
        )

    def start(self, img):
        # Document extraction
        # img1 = self.preproc(img)

        if self.debug:
            saved_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.preprocess_cache, saved_img)

            boxes, img2 = self.det_model(
                img,
                crop_region=True,
                return_result=True,
                output_path=self.cache_folder,
            )
            saved_img = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.detection_cache, saved_img)
        else:
            boxes = self.det_model(
                img,
                crop_region=True,
                return_result=False,
                output_path=self.cache_folder,
            )

        img_paths = os.listdir(self.crop_cache)
        img_paths.sort(key=natural_keys)
        img_paths = [os.path.join(self.crop_cache, i) for i in img_paths]

        texts, scores = self.ocr_model.predict_folder(img_paths, return_probs=True)
        texts = self.correction(texts, return_score=False)

        return [
            [box.tolist(), (text, score)]
            for box, text, score in zip(boxes, texts, scores)
        ]


if __name__ == "__main__":
    config = Config("./tool/config/configs.yaml")
    args = init_args()
    pipeline = Pipeline(args, config)
    img = cv2.imread(
        "/app/vietnamese-ocr-toolbox/demo/data_samples/mcocr_public_145013atlmq.jpg"
    )
    start_time = time.time()
    res = pipeline.start(img)
    print(res)
    end_time = time.time()

    print(f"Executed in {end_time - start_time} s")

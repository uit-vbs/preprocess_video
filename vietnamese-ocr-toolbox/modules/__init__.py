import os
import cv2
import shutil
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import mapping, MultiPoint
from .preprocess import DocScanner
import modules.ocr as ocr
import modules.correction as correction
from tool.config import Config
from tool.utils import download_pretrained_weights
from modules.detection.predict import crop_box

from mmocr.utils.ocr import MMOCR
from paddleocr import PaddleOCR, draw_ocr


CACHE_DIR = ".cache"


class Preprocess:
    def __init__(self, find_best_rotation=True, det_model=None, ocr_model=None):

        self.find_best_rotation = find_best_rotation

        if self.find_best_rotation:
            self.crop_path = os.path.join(CACHE_DIR, "crops")
            if os.path.exists(self.crop_path):
                shutil.rmtree(self.crop_path)
                os.mkdir(self.crop_path)
            self.det_model = det_model if det_model is not None else Detection()
            self.ocr_model = ocr_model if ocr_model is not None else OCR()
        self.scanner = DocScanner()
        

    def __call__(self, image, return_score=False):

        output = self.scanner.scan(image)

        if self.find_best_rotation:

            _ = self.det_model(
                output, crop_region=True, return_result=False, output_path=CACHE_DIR
            )

            orientation_scores = np.array([0.0, 0.0, 0.0, 0.0])
            num_crops = len(os.listdir(self.crop_path))
            for i in range(num_crops):
                single_crop_path = os.path.join(self.crop_path, f"{i}.jpg")
                if not os.path.isfile(single_crop_path):
                    continue
                img = cv2.imread(single_crop_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                orientation_scores += ocr.find_rotation_score(img, self.ocr_model)
            best_orient = np.argmax(orientation_scores)
            print(f"Rotate image by {best_orient*90} degrees")

            # Rotate the original image
            output = ocr.rotate_img(output, best_orient)

        return (output, orientation_scores) if return_score else output


class Detection:
    def __init__(self, det_model_name=None, config_dir=None, device=None):
        self.det_model_name = det_model_name
        self.config_dir = config_dir
        self.device = device

        self.model = MMOCR(det=self.det_model_name, use_gpu=self.config_dir, recog=None, device=self.device, batch_mode=True)

    def __call__(self, image, crop_region=False, return_result=False, output_path=None):
        """
        Input: path to image
        Output: boxes (coordinates of 4 points)
        """

        if output_path is None:
            assert crop_region, "Please specify output_path"
        else:
            output_path = os.path.join(output_path, "crops")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
                os.mkdir(output_path)

        # Detect for final result
        boxes_list = self.model.readtext(image)
        
        boxes_list = np.array([[[box[i], box[i + 1]] for i in range(0, len(box) - 1, 2)] for box in boxes_list[0]['boundary_result']])
        boxes_list = self.convert_poly_to_4point(boxes_list)
        boxes_list = crop_box(image, boxes_list, output_path)

        if return_result:
            img = draw_ocr(image, boxes_list)
            img = Image.fromarray(img)
        return (boxes_list, img) if return_result else boxes_list

    @staticmethod
    def convert_poly_to_4point(point_array):
        converted_points = []
        for point in point_array:
            minium_points = mapping(MultiPoint(point).minimum_rotated_rectangle)
            minium_points = np.array(list(map(np.array, minium_points['coordinates'][0])))
            converted_points.append(minium_points)
        
        return converted_points

class OCR:
    def __init__(self, config_path=None, weight_path=None, model_name=None, device=None):
        if config_path is None:
            config_path = "tool/config/ocr/configs.yaml"
        config = Config(config_path)
        ocr_config = ocr.Config.load_config_from_name(config.model_name)
        ocr_config["cnn"]["pretrained"] = False
        ocr_config["device"] = device
        ocr_config["predictor"]["beamsearch"] = False

        self.model_name = model_name
        if weight_path is None:
            if self.model_name is None:
                self.model_name = "transformerocr_default_vgg"
            tmp_path = os.path.join(CACHE_DIR, f"{self.model_name}.pth")
            download_pretrained_weights(self.model_name, cached=tmp_path)
            weight_path = tmp_path
        ocr_config["weights"] = weight_path
        self.model = ocr.Predictor(ocr_config)

    def __call__(self, img, return_prob=False):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return self.model.predict(img, return_prob)

    def predict_folder(self, img_paths, return_probs=False):
        texts = []
        if return_probs:
            probs = []
        for i, img_path in enumerate(img_paths):
            img = Image.open(img_path)
            if return_probs:
                text, prob = self(img, True)
                texts.append(text)
                probs.append(prob)
            else:
                text = self(img, False)
                texts.append(text)

        return (texts, probs) if return_probs else texts




class Correction:
    def __init__(self, dictionary=None, mode="ed"):
        assert mode in ["trie", "ed"], "Mode is not supported"
        self.mode = mode
        self.dictionary = dictionary

        self.use_trie = False
        self.use_ed = False

        if self.mode == "ed":
            self.use_ed = True

        elif self.mode == "trie":
            self.use_trie = True
        if self.use_ed:
            self.ed = correction.get_heuristic_correction("diff")
        if self.use_trie:
            self.trie = correction.get_heuristic_correction("trie")

        if (self.use_ed or self.use_trie) and self.dictionary is None:
            self.dictionary = {}
            df = pd.read_csv("./modules/retrieval/heuristic/custom-dictionary.csv")
            for id, row in df.iterrows():
                self.dictionary[row.text.lower()] = row.lbl

    def __call__(self, query_texts, return_score=False):
        if self.use_ed:
            preds, score = self.ed(query_texts, self.dictionary)

        if self.use_trie:
            preds, score = self.trie(query_texts, self.dictionary)

        return (preds, score) if return_score else preds

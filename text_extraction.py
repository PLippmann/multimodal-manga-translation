from abc import ABC, abstractmethod
from typing import List
import sys
from PIL import Image as Img
from pyclustering.cluster.optics import optics
from manga_ocr import MangaOcr
import json
sys.path.insert(0, '../Manga-Text-Segmentation/code/')

from fastai.vision import *
from dataset import *
from transformers import AutoModel
import numpy as np
import torch


class TextExtractor(ABC):

    @abstractmethod
    def extract_lines(self, path_to_img: str) -> List[str]:
        pass 

class TextExtractorPixelwise(TextExtractor):

    MIN_TEXT_SIZE = 15

    def __init__(self, path_to_model: str):
        file = path_to_model.split('/')[-1]
        path = path_to_model[0:-len(file)]
        self.text_segmentation_model = load_learner(path, file)
        self.mocr = MangaOcr()
        self.magi = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).cuda()

    def extract_lines(self, path_to_img: str) -> List[str]:
        img = open_image(path_to_img)

        # Resize the image if sizes not divisible by 2
        size_w, size_h = img.size
        size_w -= size_w % 2
        size_h -= size_h % 2
        img = Image(img.data[:, 0:size_w, 0:size_h])

        # Get pixelwise character position prediction
        pred = self.text_segmentation_model.predict(img)[0]
        mask = unpad_tensor(pred.px, img.px.shape)[0]

        # Cluster text from the same speech bubbles
        sample = mask.nonzero().tolist()

        if len(sample) == 0:
            return []

        radius = 30
        neighbours = 2
        optics_instance = optics(sample, radius, neighbours)
        optics_instance.process() 
        clusters_ind = optics_instance.get_clusters()
        clusters = [[sample[i] for i in indexes] for indexes in clusters_ind]

        # Get a bounding box for each text cluster
        text_boxes = [self.bounding_box(c) for c in clusters]

        # Get a bounding box for each panel
        image_magi = self.read_image_as_np_array(path_to_img)

        with torch.no_grad():
            results = self.magi.predict_detections_and_associations([image_magi])

        panel_boxes = results[0]['panels']

        # Sort the text boxes in reading order
        sorted_text_indices = self.magi.sort_panels_and_text_bboxes_in_reading_order([panel_boxes], [text_boxes])[1][0]
        text_boxes_sorted = [text_boxes[i] for i in sorted_text_indices]

        # Extract the text from each of the bounding boxes
        img_pil = Img.open(path_to_img)
        result = []
        for box in text_boxes_sorted:
            (x1, y1, x2, y2) = box

            if min(abs(x2-x1), abs(y2-y1)) < self.MIN_TEXT_SIZE:
                continue

            cropped_img = img_pil.crop((x1, y1, x2, y2))
            text = self.mocr(cropped_img)

            result.append(text)

        return result

    @classmethod
    def bounding_box(cls, points):
        y_coordinates, x_coordinates = zip(*points)

        return (min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates))
    
    @classmethod
    def read_image_as_np_array(cls, image_path):
        with open(image_path, "rb") as file:
            image = Img.open(file).convert("L").convert("RGB")
            image = np.array(image)
        return image

class OpenMantraAnnotationMatcher(TextExtractor):
    book_titles = {
        "tojime_no_siora": 0,
        "balloon_dream": 1, 
        "tencho_isoro": 2, 
        "boureisougi": 3, 
        "rasetugari": 4, 
    }

    def __init__(self, annotation_file_path: str):
        f = open(annotation_file_path)
        self.annotation_data = json.load(f)

    def extract_lines(self, path_to_img: str) -> List[str]:
        #img_path = "../open-mantra-dataset/" + page['image_paths']['ja']
        path_steps = path_to_img.split('/')
        if path_steps[-4] != 'images' or (path_steps[-3] not in self.book_titles):
            raise ValueError

        manga_index = self.book_titles[path_steps[-3]]

        page_index = int(path_steps[-1][0:-4])

        jp_lines = [text_box['text_ja'] for text_box in self.annotation_data[manga_index]['pages'][page_index]['text']]

        return jp_lines
    
class LoveHinaAnnotationMatcher(TextExtractor):

    def __init__(self, annotation_file_path: str):
        f = open(annotation_file_path)
        self.annotation_data = json.load(f)

    def extract_lines(self, path_to_img: str) -> List[str]:
        #img_path = "../open-mantra-dataset/" + page['image_paths']['ja']
        path_steps = path_to_img.split('/')
        if path_steps[-2][0:8] != 'LoveHina':
            raise ValueError

        page_index = int(path_steps[-1][0:-4])

        jp_lines = [text_box['text_jp'] for text_box in self.annotation_data[page_index]['lines']]

        return jp_lines
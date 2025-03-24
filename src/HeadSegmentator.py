import os
import cv2
import dotenv
from ultralytics import YOLO
import random
import numpy as np

class HeadSegmentator:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.classes = list(self.model.names.values())
        self.classes_ids = [self.classes.index(c) for c in self.classes]

    def segment(self, image_path, confidence=0.2):
        image = cv2.imread(image_path)
        results = self.model.predict(image, conf=confidence)
        return results

    def draw_segmentation(self, image_path, results):
        image = cv2.imread(image_path)
        colors = [(0, 0, 255)]
        for result in results:
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                cv2.fillPoly(image, points, colors[0])
        return image

    def save_image(self, image, save_path):
        cv2.imwrite(save_path, image)
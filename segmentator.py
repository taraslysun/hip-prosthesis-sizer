from ultralytics import YOLO
import cv2
import random
import numpy as np
import os

class Segmentator:
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
        colors = [random.choices(range(256), k=3) for _ in self.classes_ids]
        for result in results:
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                color_number = self.classes_ids.index(int(box.cls[0]))
                cv2.fillPoly(image, points, colors[color_number])
        return image
    
    def save_image(self, image, save_path):
        cv2.imwrite(save_path, image)



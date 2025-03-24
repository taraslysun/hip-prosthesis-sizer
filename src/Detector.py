
from ultralytics import YOLO
import cv2
import random
import numpy as np

class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.classes = list(self.model.names.values())

    def detect(self, image_path, confidence=0.2):
        image = cv2.imread(image_path)
        results = self.model.predict(image, conf=confidence)
        return results

    def draw_detection(self, image_path, results):
        img = cv2.imread(image_path)
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = self.classes[class_id]
                confidence = float(box.conf[0])
                color = [random.randint(0, 255) for _ in range(3)]
                if class_name == 'big-hole':
                    continue
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def save_image(self, image, save_path):
        cv2.imwrite(save_path, image)

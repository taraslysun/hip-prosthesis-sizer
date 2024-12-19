from ultralytics import YOLO
import cv2
import random
import numpy as np
import dotenv
import os
from inference_sdk import InferenceHTTPClient

class Detector:
    def __init__(self, model_id):
        self.model_id = model_id
        dotenv.load_dotenv(dotenv.find_dotenv())
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=os.getenv("ROBOFLOW_API_KEY"),
        )

    def detect(self, image_path):
        results = self.client.infer(image_path, model_id=self.model_id)
        return results['predictions']
    
    def draw_detection(self, image_path, results):
        img = cv2.imread(image_path)
        for result in results:
            if (result['class'] == 'big-hole'):
                continue
            x = float(result['x'])
            y = float(result['y'])
            w = float(result['width'])
            h = float(result['height'])
            color = random.choices(range(256), k=3)
            top_left = (int(x - w / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(img, top_left, bottom_right, color, 1)

            cv2.putText(img, result['class'], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        return img
    
    def save_image(self, image, save_path):
        cv2.imwrite(save_path, image)

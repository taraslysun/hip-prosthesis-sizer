import os
import cv2
import dotenv
from inference_sdk import InferenceHTTPClient

class HeadSegmentator:
    def __init__(self, model_id):
        self.model_id = model_id
        dotenv.load_dotenv(dotenv.find_dotenv())
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=os.getenv("ROBOFLOW_API_KEY"),
        )

    def segment(self, image_path):      
        results = self.client.infer(image_path, model_id=self.model_id)
        return results['predictions']
    
    def draw_segmentation(self, image_path, results):
        image = cv2.imread(image_path)
        for result in results:
            for point in result['points']:
                x = int(point['x'])
                y = int(point['y'])
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        return image
    

    def save_image(self, image, save_path):
        cv2.imwrite(save_path, image)


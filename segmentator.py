from ultralytics import YOLO
import cv2
import random
import numpy as np
import os
import dotenv
from inference_sdk import InferenceHTTPClient

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




def find_tangent_lines(image_path, segmentation_results, detection_results):
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for result in segmentation_results:
        for seg in result.masks.xy:
            points = np.int32([seg])
            cv2.fillPoly(mask, points, 255)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found in segmentation.")
        return image

    green_contour = max(contours, key=cv2.contourArea)

    small_holes = []
    for result in detection_results:
        if result['class'] == 'small-hole':
            x, y, w, h = result['x'], result['y'], result['width'], result['height']
            top_right = (int(x + w / 2), int(y - h / 2))
            top_left = (int(x - w / 2), int(y - h / 2))
            small_holes.append((top_left, top_right))

    if len(small_holes) < 2:
        print("Not enough 'small-hole' detections to draw tangent lines.")
        return image

    left_hole = min(small_holes, key=lambda x: x[0][0])
    right_hole = max(small_holes, key=lambda x: x[0][0])
    left_hole_top_left, left_hole_top_right = left_hole
    right_hole_top_left, right_hole_top_right = right_hole


    def line_intersects_contour(point1, point2, contour):
        mask_line = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.line(mask_line, point1, point2, 255, 1)
        mask_contour = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_contour, [contour], -1, 255, 1)
        intersection = cv2.bitwise_and(mask_line, mask_contour)
        return np.any(intersection)
    

    new_points = []
    left_line_points = None
    for angle in range(0, 90):
        length = 1000
        theta = np.radians(180 - angle)
        x_offset = int(length * np.cos(theta))
        y_offset = int(length * np.sin(theta))
        new_point = (left_hole_top_left[0] + x_offset, left_hole_top_left[1] - y_offset)
        if line_intersects_contour(left_hole_top_left, new_point, green_contour):
            new_points.append(new_point)
            left_line_points = (left_hole_top_left, new_point)
            print(left_line_points, "left")
            cv2.line(image, left_hole_top_left, new_point, (255, 0, 0), 2)
            break

    right_line_points = None
    for angle in range(0, 90):
        length = 1000
        theta = np.radians(angle)
        x_offset = int(length * np.cos(theta))
        y_offset = int(length * np.sin(theta))
        new_point = (right_hole_top_right[0] + x_offset, right_hole_top_right[1] - y_offset)
        if line_intersects_contour(right_hole_top_right, new_point, green_contour):
            new_points.append(new_point)
            right_line_points = (right_hole_top_right, new_point)
            cv2.line(image, right_hole_top_right, new_point, (0, 255, 0), 2)
            break

    return image, left_line_points, right_line_points, left_hole_top_left, right_hole_top_right, new_points


def get_head_edge_points(head_segm_results):
    res = []
    for points in head_segm_results:
        max_left = sorted([point for point in points['points']], key=lambda x: x['x'])[-1]
        max_right = sorted([point for point in points['points']], key=lambda x: x['x'])[0]
        res.append((max_left, max_right))
    return res


def px_in_cm(det_results):
    pixels = None
    for res in det_results:
        if res['class'] == 'scale-mark':
            pixels = res['width']
            break
    return pixels * 0.69



def get_head_highest_points(head_segm_results, cm):
    res = []
    for result in head_segm_results:
        highest_point = min(result['points'], key=lambda x: x['y'])
        print(highest_point)
        res.append((highest_point['x'], highest_point['y']-0.5*cm))

    return res


if __name__ == "__main__":
    image1 = "test_data/img1.jpg"
    image2 = "test_data/img2.jpg"
    image3 = "test_data/img3.jpg"
    head_segm = HeadSegmentator(model_id="femur-head-segmentation/2")

    res_head_segm = head_segm.segment(image3)
    img_head_segm = head_segm.draw_segmentation(image3, res_head_segm)
    print(head_segm.get_highest_points(res_head_segm))
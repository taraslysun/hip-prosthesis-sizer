from ultralytics import YOLO
import cv2
import random
import numpy as np
import os
import dotenv
from inference_sdk import InferenceHTTPClient
from detector import Detector

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
    
    left_hole_top_left, _ = left_hole
    _, right_hole_top_right = right_hole

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

    res = {
        'image': image,
        'left_line_points': left_line_points,
        'right_line_points': right_line_points,
        'left_hole_top_left': left_hole_top_left,
        'right_hole_top_right': right_hole_top_right,
        'new_points': new_points
    }

    return res


def get_head_edge_points(head_segm_results):
    res = []
    for result in head_segm_results:
        for point in result['points']:
            res.append((point['x'], point['y']))
    max_right = max(res, key=lambda x: x[0])
    max_left = min(res, key=lambda x: x[0])
    return max_left, max_right


def px_in_cm(det_results, offset=0.8):
    pixels = None
    for res in det_results:
        if res['class'] == 'scale-mark':
            pixels = res['width']
            break
    return pixels * offset



def get_head_highest_points(head_segm_results, cm, offset=0.5):
    res = []
    for result in head_segm_results:
        highest_point = min(result['points'], key=lambda x: x['y'])
        res.append((highest_point['x'], highest_point['y']-offset*cm))
    
    left_highest = min(res, key=lambda x: x[0])
    right_highest = max(res, key=lambda x: x[0])
    return left_highest, right_highest

def get_tang_points(L_highest_point, R_highest_point, L_right_highest_point, R_left_highest_point, L_hole_top_left, R_hole_top_right):
    x0_left, y0_left = L_right_highest_point[0], L_right_highest_point[1]
    x1_left, y1_left = L_hole_top_left[0], L_hole_top_left[1]
    y_left = L_highest_point[1]

    x_intersection_left = x0_left + ((y_left - y0_left) * (x1_left - x0_left)) / (y1_left - y0_left)
    
    x0_right, y0_right = R_left_highest_point[0], R_left_highest_point[1]
    x1_right, y1_right = R_hole_top_right[0], R_hole_top_right[1]
    y_right = R_highest_point[1]

    x_intersection_right = x0_right + ((y_right - y0_right) * (x1_right - x0_right)) / (y1_right - y0_right)

    L_right_top = (int(x_intersection_left), int(y_left))
    R_left_top = (int(x_intersection_right), int(y_right))

    return L_right_top, R_left_top


def get_keypoints(image_path, segm_model, head_segm_model, det_model):
    segm = Segmentator(model_path=segm_model)
    head_segm = HeadSegmentator(model_id=head_segm_model)
    det = Detector(model_id=det_model)

    segm_results = segm.segment(image_path)
    head_segm_results = head_segm.segment(image_path)
    det_results = det.detect(image_path)

    img = cv2.imread(image_path)

    tang_dict = find_tangent_lines(image_path, segm_results, det_results)
    cm = px_in_cm(det_results)

    new_points = tang_dict['new_points']
    L_hole_top_left = tang_dict['left_hole_top_left']
    R_hole_top_right = tang_dict['right_hole_top_right']

    L_edge_point, R_edge_point = get_head_edge_points(head_segm_results)
    L_highest_point, R_highest_point = get_head_highest_points(head_segm_results, cm)
    L_right_highest_point = new_points[0]
    R_left_highest_point = new_points[1]

    L_left_top = (int(L_edge_point[0]), int(L_highest_point[1]))
    L_left_bottom = (int(L_edge_point[0]), int(L_hole_top_left[1]))
    L_right_bottom = (int(L_hole_top_left[0]), int(L_hole_top_left[1]))


    R_right_top = (int(R_edge_point[0]), int(R_highest_point[1]))
    R_right_bottom = (int(R_edge_point[0]), int(R_hole_top_right[1]))
    R_left_bottom = (int(R_hole_top_right[0]), int(R_hole_top_right[1]))

    L_right_top, R_left_top = get_tang_points(L_highest_point, 
                                              R_highest_point, 
                                              L_right_highest_point, 
                                              R_left_highest_point, 
                                              L_hole_top_left, 
                                              R_hole_top_right)



    return img, {
        'L_edge_point': L_edge_point,
        'R_edge_point': R_edge_point,
        'L_highest_point': L_highest_point,
        'R_highest_point': R_highest_point,
        'L_left_top': L_left_top,
        'L_left_bottom': L_left_bottom,
        'L_right_top': L_right_top,
        'L_right_bottom': L_right_bottom,
        'R_right_top': R_right_top,
        'R_right_bottom': R_right_bottom,
        'R_left_top': R_left_top,
        'R_left_bottom': R_left_bottom,
        'cm': cm
    }


if __name__ == "__main__":
    image1 = "test_data/img1.jpg"
    image2 = "test_data/img2.jpg"
    image3 = "test_data/img3.jpg"
    head_segm = HeadSegmentator(model_id="femur-head-segmentation/2")

    res_head_segm = head_segm.segment(image3)
    img_head_segm = head_segm.draw_segmentation(image3, res_head_segm)
    print(head_segm.get_highest_points(res_head_segm))
import numpy as np
import cv2
from src.Segmentator import Segmentator
from src.HeadSegmentator import HeadSegmentator
from src.Detector import Detector



class HipJointDetector:
    def __init__(self, segm_model_path, head_segm_model_path, det_model_path):
        self.segm = Segmentator(model_path=segm_model_path)
        self.head_segm = HeadSegmentator(model_path=head_segm_model_path)
        self.det = Detector(model_path=det_model_path)


    def find_tangent_lines(self, image_path, segmentation_results, detection_results):
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


    def get_head_edge_points(self, head_segm_results):
        res = []
        for result in head_segm_results:
            for point in result['points']:
                res.append((point['x'], point['y']))
        max_right = max(res, key=lambda x: x[0])
        max_left = min(res, key=lambda x: x[0])
        return max_left, max_right


    def px_in_cm(self, det_results, scale=0.8):
        pixels = None
        for res in det_results:
            if res['class'] == 'scale-mark':
                pixels = res['width']
                break
        return pixels * scale


    def get_head_highest_points(self, head_segm_results, cm, offset=0.5):
        res = []
        for result in head_segm_results:
            highest_point = min(result['points'], key=lambda x: x['y'])
            res.append((highest_point['x'], highest_point['y']-offset*cm))
        
        left_highest = min(res, key=lambda x: x[0])
        right_highest = max(res, key=lambda x: x[0])
        return left_highest, right_highest


    def get_tang_points(self, 
                        L_highest_point, 
                        R_highest_point, 
                        L_right_highest_point, 
                        R_left_highest_point, 
                        L_hole_top_left, 
                        R_hole_top_right):
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


    def get_keypoints(self, image_path):

        segm_results = self.segm.segment(image_path)
        head_segm_results = self.head_segm.segment(image_path)
        det_results = self.det.detect(image_path)

        img = cv2.imread(image_path)

        tang_dict = self.find_tangent_lines(image_path, segm_results, det_results)
        cm = self.px_in_cm(det_results)

        new_points = tang_dict['new_points']
        L_hole_top_left = tang_dict['left_hole_top_left']
        R_hole_top_right = tang_dict['right_hole_top_right']

        L_edge_point, R_edge_point = self.get_head_edge_points(head_segm_results)
        L_highest_point, R_highest_point = self.get_head_highest_points(head_segm_results, cm)
        L_right_highest_point = new_points[0]
        R_left_highest_point = new_points[1]

        L_left_top = (int(L_edge_point[0]), int(L_highest_point[1]))
        L_left_bottom = (int(L_edge_point[0]), int(L_hole_top_left[1]))
        L_right_bottom = (int(L_hole_top_left[0]), int(L_hole_top_left[1]))


        R_right_top = (int(R_edge_point[0]), int(R_highest_point[1]))
        R_right_bottom = (int(R_edge_point[0]), int(R_hole_top_right[1]))
        R_left_bottom = (int(R_hole_top_right[0]), int(R_hole_top_right[1]))

        L_right_top, R_left_top = self.get_tang_points(L_highest_point, 
                                                R_highest_point, 
                                                L_right_highest_point, 
                                                R_left_highest_point, 
                                                L_hole_top_left, 
                                                R_hole_top_right)


        points = {
            'L_left_top': L_left_top,
            'L_left_bottom': L_left_bottom,
            'L_right_top': L_right_top,
            'L_right_bottom': L_right_bottom,
            'R_right_top': R_right_top,
            'R_right_bottom': R_right_bottom,
            'R_left_top': R_left_top,
            'R_left_bottom': R_left_bottom,
        }
        return img, points, cm


    def draw_detected_points(self, image_path, detector):
        """
        Draws all detected key points on the image.
        
        :param image_path: Path to the input image.
        :param detector: An instance of HipJointDetector.
        :return: Image with drawn key points.
        """
        img, points, _ = self.get_keypoints(image_path)
        
        # Define colors for the points
        color = (0, 0, 255)  # Red
        radius = 5  # Size of the points
        thickness = -1  # Fill the circle
        
        for point_name, coordinates in points.items():
            cv2.circle(img, coordinates, radius, color, thickness)
            cv2.putText(img, point_name, (coordinates[0] + 5, coordinates[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img

import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv11 model (assuming it's in PyTorch format)
def load_model(model_path):
    model = YOLO(model_path)  # Load model
    return model

# Perform inference and draw segmentation masks
def infer_and_draw(model, image_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    orig_image = image.copy()
    height, width, _ = image.shape

    # Preprocess image (resize, normalize, etc., based on your model's requirements)
    input_size = (640, 640)  # Replace with the model's expected input size
    image_resized = cv2.resize(image, input_size)
    image_normalized = image_resized / 255.0  # Normalize to [0, 1]
    image_transposed = np.transpose(image_normalized, (2, 0, 1))  # Channels first
    image_tensor = torch.tensor(image_transposed, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    print(f"Detected {len(predictions)} objects")
    print(predictions)

    # Post-process predictions (Assuming YOLO-like output format with masks)
    for pred in predictions:
        boxes = pred['boxes']  # Bounding boxes
        masks = pred['masks']  # Segmentation masks

        for box, score, mask in zip(boxes, scores, masks):
            if score < 0.5:  # Confidence threshold
                continue

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw segmentation mask
            mask = mask.squeeze().cpu().numpy()  # Convert to numpy
            mask = cv2.resize(mask, (width, height))  # Resize to original image size
            mask = (mask > 0.5).astype(np.uint8)  # Binary mask

            # Apply mask on image
            colored_mask = np.zeros_like(orig_image)
            colored_mask[:, :, 1] = mask * 255  # Green mask
            orig_image = cv2.addWeighted(orig_image, 1, colored_mask, 0.5, 0)

    # Save the result
    cv2.imwrite(output_path, orig_image)

# Usage
if __name__ == "__main__":
    model_path = "segmentation.pt"  # Update with your model path
    image_path = "image.jpg"   # Update with your image path
    output_path = "out.jpg" # Update with desired output path

    model = load_model(model_path)
    infer_and_draw(model, image_path, output_path)

    print(f"Processed image saved to {output_path}")

# filepath: /home/david/projs/HSCa_2025_demos/yolov8_transfer_learning/evaluate.py

import os
import cv2
import numpy as np
from ultralytics import YOLO
from glob import glob

# Load models
custom_model = YOLO("yolov8s_front_rears.pt")
coco_model = YOLO("yolov8s_COCO.pt")

# Class information
custom_classes = {0: "Vehicle Front", 1: "Vehicle Rear"}
coco_vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 6: "train"}

# Colors for visualization (BGR format)
colors = {
    "front": (0, 255, 0),     # Green for vehicle front
    "rear": (0, 0, 255),      # Red for vehicle rear
    "coco_vehicle": (255, 0, 0)  # Blue for COCO vehicle classes
}

# Get image paths
image_dir = "images"
image_paths = sorted(glob(os.path.join(image_dir, "*")))

if not image_paths:
    print(f"No images found in {image_dir} directory!")
    exit()

# Initialize image index
img_idx = 0

while True:
    # Load the image
    img_path = image_paths[img_idx]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        img_idx = (img_idx + 1) % len(image_paths)
        continue
    
    # Create a copy for visualization
    vis_img = img.copy()
    
    # Run inference with custom model
    custom_results = custom_model(img, verbose=False)[0]
    
    # Process custom model predictions
    for det in custom_results.boxes:
        cls_id = int(det.cls.item())
        if cls_id in custom_classes:
            box = det.xyxy[0].cpu().numpy().astype(int)
            conf = float(det.conf.item())
            
            # Select color based on class
            color = colors["front"] if cls_id == 0 else colors["rear"]
            
            # Draw bounding box
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Label
            label = f"{custom_classes[cls_id]}: {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            c2 = box[0] + t_size[0] + 3, box[1] + t_size[1] + 4
            cv2.rectangle(vis_img, (box[0], box[1]), c2, color, -1)
            cv2.putText(vis_img, label, (box[0], box[1] + t_size[1] + 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Run inference with COCO model
    coco_results = coco_model(img, verbose=False)[0]
    
    # Process COCO model predictions
    for det in coco_results.boxes:
        cls_id = int(det.cls.item())
        if cls_id in coco_vehicle_classes:
            box = det.xyxy[0].cpu().numpy().astype(int)
            conf = float(det.conf.item())
            
            # Draw bounding box
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), colors["coco_vehicle"], 2)
            
            # Label
            label = f"{coco_vehicle_classes[cls_id]}: {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            c2 = box[0] + t_size[0] + 3, box[1] + t_size[1] + 4
            cv2.rectangle(vis_img, (box[0], box[1]), c2, colors["coco_vehicle"], -1)
            cv2.putText(vis_img, label, (box[0], box[1] + t_size[1] + 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display image index
    index_text = f"{img_idx + 1}/{len(image_paths)}"
    t_size = cv2.getTextSize(index_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.putText(vis_img, index_text, 
                (vis_img.shape[1] - t_size[0] - 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(vis_img, index_text, 
                (vis_img.shape[1] - t_size[0] - 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add legend
    legend_y = 70
    for class_name, color_name in [("Vehicle Front", "front"), 
                                   ("Vehicle Rear", "rear"),
                                   ("COCO Vehicles", "coco_vehicle")]:
        cv2.rectangle(vis_img, (vis_img.shape[1] - 200, legend_y), 
                     (vis_img.shape[1] - 180, legend_y + 20), 
                     colors[color_name], -1)
        cv2.putText(vis_img, class_name, 
                    (vis_img.shape[1] - 175, legend_y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 30
    
    # Display in full screen
    cv2.namedWindow("YOLOv8 Comparison", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("YOLOv8 Comparison", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("YOLOv8 Comparison", vis_img)
    
    # Handle keyboard input
    key = cv2.waitKey(0) & 0xFF
    
    # Navigation
    if key == ord('d'):  # Next image
        img_idx = (img_idx + 1) % len(image_paths)
    elif key == ord('a'):  # Previous image
        img_idx = (img_idx - 1) % len(image_paths)
    # Exit
    elif key == ord('q') or key == 27:  # 'q' or ESC
        break

cv2.destroyAllWindows()
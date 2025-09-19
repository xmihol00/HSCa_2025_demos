import os
import cv2
import numpy as np
import glob
from pathlib import Path

class YOLOAnnotator:
    def __init__(self):
        # Initialize directories
        self.images_dir = 'images'
        self.output_dir = 'fronts_rears_dataset'
        self.labels_dir = os.path.join(self.output_dir, 'labels')
        self.images_output_dir = os.path.join(self.output_dir, 'images')
        
        # Create output directories if they don't exist
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.images_output_dir, exist_ok=True)
        
        # Get list of images
        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')) + 
                                 glob.glob(os.path.join(self.images_dir, '*.jpeg')) +
                                 glob.glob(os.path.join(self.images_dir, '*.png')))
        
        if not self.image_paths:
            print(f"No images found in {self.images_dir}")
            exit(1)
            
        # Class definitions
        self.class_names = ['vehicle_front', 'vehicle_rear']
        self.class_colors = [(0, 255, 0), (0, 0, 255)]  # Green for front, Red for rear
        
        # Current state
        self.current_img_idx = 0
        self.current_class_idx = 0
        self.mode = "draw"  # "draw" or "edit"
        self.bboxes = []  # List of [class_idx, x_center, y_center, width, height] in relative coordinates
        self.drawing = False
        self.editing = False
        self.edit_bbox_idx = -1
        self.edit_edge = None  # "top", "bottom", "left", "right"
        self.start_point = (0, 0)
        self.end_point = (0, 0)
        self.edge_sensitivity = 25  # pixels - increased for easier edge selection
        
        # Window setup
        cv2.namedWindow('YOLOAnnotator', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('YOLOAnnotator', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('YOLOAnnotator', self.handle_mouse_events)
        
        # Load first image
        self.load_image()
        
    def load_image(self):
        self.img_path = self.image_paths[self.current_img_idx]
        self.img = cv2.imread(self.img_path)
        self.img_height, self.img_width = self.img.shape[:2]
        
        # Generate label path
        img_filename = os.path.basename(self.img_path)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        self.label_path = os.path.join(self.labels_dir, label_filename)
        
        # Load existing annotations if available
        self.bboxes = []
        if os.path.exists(self.label_path):
            with open(self.label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_idx = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            self.bboxes.append([class_idx, x_center, y_center, width, height])
        
        self.display_image()
    
    def save_annotations(self):
        with open(self.label_path, 'w') as f:
            for bbox in self.bboxes:
                f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
        
        # Save image to output directory
        img_filename = os.path.basename(self.img_path)
        output_img_path = os.path.join(self.images_output_dir, img_filename)
        cv2.imwrite(output_img_path, self.img)
        
        print(f"Saved annotations to {self.label_path}")
    
    def relative_to_absolute(self, bbox):
        """Convert relative YOLO coordinates to absolute pixel coordinates"""
        class_idx, x_center, y_center, width, height = bbox
        
        x_min = int((x_center - width / 2) * self.img_width)
        y_min = int((y_center - height / 2) * self.img_height)
        x_max = int((x_center + width / 2) * self.img_width)
        y_max = int((y_center + height / 2) * self.img_height)
        
        return class_idx, x_min, y_min, x_max, y_max
    
    def absolute_to_relative(self, class_idx, x_min, y_min, x_max, y_max):
        """Convert absolute pixel coordinates to relative YOLO coordinates"""
        x_center = (x_min + x_max) / (2 * self.img_width)
        y_center = (y_min + y_max) / (2 * self.img_height)
        width = (x_max - x_min) / self.img_width
        height = (y_max - y_min) / self.img_height
        
        return [class_idx, x_center, y_center, width, height]
    
    def get_nearest_bbox_and_edge(self, x, y):
        """Find the nearest bounding box and edge to the given point"""
        min_dist = float('inf')
        nearest_idx = -1
        nearest_edge = None
        
        for i, bbox in enumerate(self.bboxes):
            class_idx, x_min, y_min, x_max, y_max = self.relative_to_absolute(bbox)
            
            # Check if point is inside bbox
            if x_min <= x <= x_max and y_min <= y <= y_max:
                # Calculate distance to each edge
                dist_top = abs(y - y_min)
                dist_bottom = abs(y - y_max)
                dist_left = abs(x - x_min)
                dist_right = abs(x - x_max)
                
                min_edge_dist = min(dist_top, dist_bottom, dist_left, dist_right)
                
                if min_edge_dist < min_dist and min_edge_dist < self.edge_sensitivity:
                    min_dist = min_edge_dist
                    nearest_idx = i
                    
                    if min_edge_dist == dist_top:
                        nearest_edge = "top"
                    elif min_edge_dist == dist_bottom:
                        nearest_edge = "bottom"
                    elif min_edge_dist == dist_left:
                        nearest_edge = "left"
                    else:
                        nearest_edge = "right"
            
        return nearest_idx, nearest_edge
    
    def is_point_inside_bbox(self, x, y, bbox_idx):
        """Check if point is inside the specified bounding box"""
        if 0 <= bbox_idx < len(self.bboxes):
            class_idx, x_min, y_min, x_max, y_max = self.relative_to_absolute(self.bboxes[bbox_idx])
            return x_min <= x <= x_max and y_min <= y <= y_max
        return False
    
    def handle_mouse_events(self, event, x, y, flags, param):
        if self.mode == "draw":
            self.handle_draw_mode(event, x, y, flags)
        elif self.mode == "edit":
            self.handle_edit_mode(event, x, y, flags)
    
    def handle_draw_mode(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self.display_image()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.end_point = (x, y)
                
                # Make sure start_point is top-left and end_point is bottom-right
                x_min = min(self.start_point[0], self.end_point[0])
                y_min = min(self.start_point[1], self.end_point[1])
                x_max = max(self.start_point[0], self.end_point[0])
                y_max = max(self.start_point[1], self.end_point[1])
                
                # Convert to relative coordinates and add to bboxes
                if x_min < x_max and y_min < y_max:  # Ensure box has positive area
                    bbox = self.absolute_to_relative(self.current_class_idx, x_min, y_min, x_max, y_max)
                    self.bboxes.append(bbox)
                    self.display_image()
    
    def handle_edit_mode(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            nearest_idx, nearest_edge = self.get_nearest_bbox_and_edge(x, y)
            if nearest_idx != -1 and nearest_edge is not None:
                self.editing = True
                self.edit_bbox_idx = nearest_idx
                self.edit_edge = nearest_edge
                self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if not self.editing:
                # Highlight nearest edge if close enough
                nearest_idx, nearest_edge = self.get_nearest_bbox_and_edge(x, y)
                if nearest_idx != -1 and nearest_edge is not None:
                    self.display_image(highlight_bbox=nearest_idx, highlight_edge=nearest_edge)
            else:
                # Update bounding box during drag
                if 0 <= self.edit_bbox_idx < len(self.bboxes):
                    class_idx, x_min, y_min, x_max, y_max = self.relative_to_absolute(self.bboxes[self.edit_bbox_idx])
                    
                    dx = x - self.start_point[0]
                    dy = y - self.start_point[1]
                    
                    if self.edit_edge == "top":
                        y_min = min(y_max - 10, y_min + dy)
                    elif self.edit_edge == "bottom":
                        y_max = max(y_min + 10, y_max + dy)
                    elif self.edit_edge == "left":
                        x_min = min(x_max - 10, x_min + dx)
                    elif self.edit_edge == "right":
                        x_max = max(x_min + 10, x_max + dx)
                    
                    self.bboxes[self.edit_bbox_idx] = self.absolute_to_relative(
                        class_idx, x_min, y_min, x_max, y_max)
                    
                    self.start_point = (x, y)
                    self.display_image(highlight_bbox=self.edit_bbox_idx, highlight_edge=self.edit_edge)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.editing:
                self.editing = False
                self.edit_bbox_idx = -1
                self.edit_edge = None
                self.display_image()
    
    def display_image(self, highlight_bbox=-1, highlight_edge=None):
        display_img = self.img.copy()
        
        # Draw all bounding boxes
        for i, bbox in enumerate(self.bboxes):
            class_idx, x_min, y_min, x_max, y_max = self.relative_to_absolute(bbox)
            color = self.class_colors[class_idx]
            
            # Draw with thicker line if this is the highlighted bbox
            thickness = 3 if i == highlight_bbox else 2
            
            # Draw bounding box
            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), color, thickness)
            
            # Draw label background
            label = self.class_names[class_idx]
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(display_img, (x_min, y_min - 30), (x_min + text_size[0], y_min), color, -1)
            
            # Draw label text
            cv2.putText(display_img, label, (x_min, y_min - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Draw highlighted edge if in edit mode
            if i == highlight_bbox and highlight_edge:
                highlight_color = (0, 255, 255)  # Cyan for highlighted edge - more visible
                edge_thickness = 4  # Thicker line for better visibility
                if highlight_edge == "top":
                    cv2.line(display_img, (x_min, y_min), (x_max, y_min), highlight_color, edge_thickness)
                elif highlight_edge == "bottom":
                    cv2.line(display_img, (x_min, y_max), (x_max, y_max), highlight_color, edge_thickness)
                elif highlight_edge == "left":
                    cv2.line(display_img, (x_min, y_min), (x_min, y_max), highlight_color, edge_thickness)
                elif highlight_edge == "right":
                    cv2.line(display_img, (x_max, y_min), (x_max, y_max), highlight_color, edge_thickness)
        
        # Draw current rectangle being created in draw mode
        if self.drawing and self.mode == "draw":
            cv2.rectangle(display_img, self.start_point, self.end_point, 
                          self.class_colors[self.current_class_idx], 2)
        
        # Add UI elements
        # Current class indicator
        class_label = f"Class: {self.class_names[self.current_class_idx]}"
        cv2.putText(display_img, class_label, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.class_colors[self.current_class_idx], 3)
        
        # Mode indicator
        mode_text = f"Mode: {self.mode.upper()}"
        text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        mode_x = (display_img.shape[1] - text_size[0]) // 2
        cv2.putText(display_img, mode_text, (mode_x, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Image counter
        img_counter = f"Image: {self.current_img_idx + 1}/{len(self.image_paths)}"
        text_size = cv2.getTextSize(img_counter, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        counter_x = display_img.shape[1] - text_size[0] - 20
        cv2.putText(display_img, img_counter, (counter_x, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Help text
        help_text = "Controls: 1-9=Class, E=Mode, A/D=Prev/Next, R=Remove, Q/ESC=Quit"
        cv2.putText(display_img, help_text, (20, display_img.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        cv2.imshow('YOLOAnnotator', display_img)
    
    def run(self):
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Navigation
            if key == ord('a'):  # Previous image
                self.save_annotations()
                self.current_img_idx = max(0, self.current_img_idx - 1)
                self.load_image()
            
            elif key == ord('d'):  # Next image
                self.save_annotations()
                self.current_img_idx = min(len(self.image_paths) - 1, self.current_img_idx + 1)
                self.load_image()
            
            # Mode toggle
            elif key == ord('e'):
                self.mode = "edit" if self.mode == "draw" else "draw"
                self.drawing = False
                self.editing = False
                self.display_image()
            
            # Class selection
            elif ord('1') <= key <= ord('9'):
                selected_class = key - ord('1')
                if selected_class < len(self.class_names):
                    self.current_class_idx = selected_class
                    self.display_image()
            
            # Remove bounding box
            elif key == ord('r'):
                if self.mode == "edit":
                    # Get mouse position
                    x, y = self.get_mouse_position()
                    if x is not None and y is not None:
                        # Find if cursor is inside any bounding box
                        for i, bbox in enumerate(self.bboxes):
                            class_idx, x_min, y_min, x_max, y_max = self.relative_to_absolute(bbox)
                            if x_min <= x <= x_max and y_min <= y <= y_max:
                                # Remove this box
                                self.bboxes.pop(i)
                                self.display_image()
                                break
            
            # Quit
            elif key == ord('q') or key == 27:  # q or ESC
                self.save_annotations()
                break
    
    def get_mouse_position(self):
        """Get current mouse position in the window"""
        # This is a workaround since OpenCV doesn't provide a direct way to get mouse position
        # We use a callback to capture the mouse position
        mouse_pos = [None, None]
        
        def get_pos(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                mouse_pos[0] = x
                mouse_pos[1] = y
        
        # Store original callback function
        self.original_callback_func = self.handle_mouse_events
        
        # Set temporary callback
        cv2.setMouseCallback('YOLOAnnotator', get_pos)
        
        # Wait a small amount of time for mouse movement
        cv2.waitKey(5)
        
        # Restore original callback
        cv2.setMouseCallback('YOLOAnnotator', self.handle_mouse_events)
        
        return mouse_pos[0], mouse_pos[1]

if __name__ == "__main__":
    annotator = YOLOAnnotator()
    annotator.run()
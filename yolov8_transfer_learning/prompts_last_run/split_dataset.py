import os
import shutil
import random
from pathlib import Path
import yaml

def split_dataset(source_images_dir="fronts_rears_dataset/images", 
                 source_labels_dir="fronts_rears_dataset/labels",
                 output_dir="dataset",
                 train_ratio=0.8):
    """
    Split dataset into training and validation sets.
    
    Args:
        source_images_dir: Directory containing images
        source_labels_dir: Directory containing labels
        output_dir: Output directory for train/val split
        train_ratio: Ratio of data to use for training (rest for validation)
    """
    # Remove output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create necessary directories
    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    val_images_dir = os.path.join(output_dir, "val", "images")
    val_labels_dir = os.path.join(output_dir, "val", "labels")
    
    for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle the files
    random.shuffle(image_files)
    
    # Calculate split index
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    
    # Copy files to their respective directories
    for file_list, img_dir, lbl_dir in [(train_files, train_images_dir, train_labels_dir),
                                      (val_files, val_images_dir, val_labels_dir)]:
        for image_file in file_list:
            # Get file name without extension
            file_base = os.path.splitext(image_file)[0]
            
            # Copy image file
            src_img = os.path.join(source_images_dir, image_file)
            dst_img = os.path.join(img_dir, image_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding label file if it exists
            label_file = f"{file_base}.txt"
            src_lbl = os.path.join(source_labels_dir, label_file)
            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(lbl_dir, label_file)
                shutil.copy2(src_lbl, dst_lbl)
    
    # Create data.yaml file
    data_yaml = {
        'train': str(Path(train_images_dir)),
        'val': str(Path(val_images_dir)),
        'nc': 2,  # Number of classes
        'names': ['vehicle front', 'vehicle rear']
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print(f"Dataset split completed. Data YAML created at {os.path.join(output_dir, 'data.yaml')}")

if __name__ == "__main__":
    split_dataset()
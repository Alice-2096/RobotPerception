import os
import cv2

# Util function to load all images from folder
def load_images_from_folder(folder_path):
    images = []
    
    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        
        # Read and append each image to the 'images' list
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)
    
    return images
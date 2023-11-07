import numpy as np
from sklearn.cluster import KMeans
from util import load_images_from_folder
from features import find_feature_points
"""
build a vocabulary of visual words from dataset
"""

def build_vocabulary(image_path, vocab_size):
    # load training images
    images = load_images_from_folder(image_path)
    if len(images) == 0:
        raise ValueError('No images found in the given folder.')
    descriptors = []
    for img in images:
        _, descriptor = find_feature_points(img)
        descriptors.append(descriptor)
    # concatenate all features to one single list 
    features = np.array(descriptors)

    # clustering
    kmeans = KMeans(n_clusters=vocab_size, random_state=0)  
    kmeans.fit(features)
    visual_words = kmeans.cluster_centers_

    # Return the cluster centers (visual words)
    return visual_words



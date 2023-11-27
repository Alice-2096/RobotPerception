import numpy as np
from sklearn.cluster import KMeans
import cv2
import os


# Util function to load all images from folder
def load_images_from_folder(folder_path):

    images = []
    # List all image files in the folder
    image_files = [f for f in os.listdir(
        folder_path) if f.endswith('.jpg') or f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Read and append each image to the 'images' list
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)

    return images


def find_feature_points(images):
    # ORB feature detection
    orb = cv2.ORB_create()
    kp = []
    des = []
    des_single_list = []
    # Detect keypoints and compute descriptors for each image
    for img in images:
        kp_img, des_img = orb.detectAndCompute(img, None)
        kp.append(kp_img)
        des.append(des_img)
        des_single_list.extend(des_img)

    return kp, des, des_single_list


# features is a single list of descriptors of all images
def build_vocabulary(features, vocab_size):
    # clustering
    kmeans = KMeans(n_clusters=vocab_size, random_state=0)
    kmeans.fit(features)
    visual_words = kmeans.cluster_centers_

    # Return the cluster centers (visual words)
    return visual_words, kmeans


# generate a list of histograms of visual words for images
def generate_feature_histograms(descriptors, kmeans):
    num_clusters = kmeans.cluster_centers_.shape[0]
    histograms = []

    for desc in descriptors:
        histogram = np.zeros(num_clusters)
        labels = kmeans.predict(desc)
        for label in labels:
            histogram[label] += 1
        histograms.append(histogram)

    return histograms

# input 1: histograph of query image
# input 2: the list of histograms of all images in the dataset


def compare_histograms(query_histogram, list_of_histograms):
    # Calculate Euclidean distances
    distances = [np.linalg.norm(query_histogram - hist)
                 for hist in list_of_histograms]

    # Find the index of the most similar histogram
    most_similar_index = np.argmin(distances)

    return most_similar_index


def process_image_and_find_best_match(kmeans, new_image, list_of_histograms):
    # Step 1: Extract features from the new image
    orb = cv2.ORB_create()
    _, descriptors = orb.detectAndCompute(new_image, None)

    # Step 2: Generate the feature histogram for the new image
    num_clusters = kmeans.cluster_centers_.shape[0]
    histogram = np.zeros(num_clusters)
    labels = kmeans.predict(descriptors)
    for label in labels:
        histogram[label] += 1

    # Step 3: Compare the histogram to the list of histograms
    most_similar_index = compare_histograms(histogram, list_of_histograms)

    # ? Alternatively, Find the indices of the 5 best candidates
    # distances = [np.linalg.norm(histogram - hist) for hist in list_of_histograms]
    # best_candidates_indices = np.argsort(distances)[:5]

    return most_similar_index

# ====================================================================================================


def train_vocab(clusters=100):
    print('Loading textures...')
    folder_path = './data/textures/'
    images = load_images_from_folder(folder_path)
    print(len(images), 'textures loaded.')

    print('Finding feature points...')
    keypoints, descriptors, descriptors_list = find_feature_points(images)
    print('Building visual dictionary...')
    visual_dictionary, kmeans = build_vocabulary(descriptors_list, clusters)
    print('Visual dictionary built.')
    return kmeans

# histogram of a single image


def generate_histogram(image, kmeans):
    orb = cv2.ORB_create()
    _, descriptors = orb.detectAndCompute(image, None)
    histogram = generate_feature_histograms([descriptors], kmeans)
    return histogram

# find best match


def best_match(target_images, kmeans, histograms):
    best_match_indices = []
    for target_image in target_images:
        print('Finding best match...')
        best_match_index = process_image_and_find_best_match(
            kmeans, target_image, histograms)
        best_match_indices.append(best_match_index)
    return best_match_indices

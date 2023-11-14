import numpy as np
from sklearn.cluster import KMeans
import cv2
from util import load_images_from_folder

def find_feature_points(images): 
    # ORB feature detection
    orb = cv2.ORB_create()
    kp = []
    des = []
    # Detect keypoints and compute descriptors for each image
    for img in images:
        kp_img, des_img = orb.detectAndCompute(img, None)
        kp.append(kp_img)
        des.append(des_img)

    return kp, des

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

"""
create a list of histograms of visual words for images 
"""
def generate_feature_histograms(descriptors, visual_dictionary):
    num_clusters = visual_dictionary.shape[0]
    histograms = []

    for desc in descriptors:
        histogram = np.zeros(num_clusters)
        labels = KMeans.predict(desc)
        for label in labels:
            histogram[label] += 1
        histograms.append(histogram)

    return histograms

# input 1: histograph of query image
# input 2: the list of histograms of all images in the dataset 
def compare_histograms(query_histogram, list_of_histograms):
    # Calculate Euclidean distances
    distances = [np.linalg.norm(query_histogram - hist) for hist in list_of_histograms]
    
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
    
    #? Alternatively, Find the indices of the 5 best candidates
    # distances = [np.linalg.norm(histogram - hist) for hist in list_of_histograms]
    # best_candidates_indices = np.argsort(distances)[:5]
    
    return most_similar_index

if __name__ == "__main__":
    folder_path = '../data/textures/'
    images = load_images_from_folder(folder_path)
    keypoints, descriptors = find_feature_points(images)
    visual_dictionary = build_vocabulary(np.vstack(descriptors), 100)
    histograms = generate_feature_histograms(descriptors, visual_dictionary)

    # load target image
    target_image = cv2.imread('../data/textures/pattern_100.png')
    # find best match
    best_match_index = process_image_and_find_best_match(visual_dictionary, target_image, histograms)
    print('Best match index: ', best_match_index)
    # show the best match
    cv2.imshow('Best match', images[best_match_index])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


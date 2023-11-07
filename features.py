import cv2
import numpy as np

def find_feature_points(image): 
    # ORB feature detection
    orb = cv2.ORB_create()
    # Detect keypoints and compute descriptors 
    kp, des = orb.detectAndCompute(image, None)

    return kp, des 

def match(kp1, des1, kp2, des2): 
    # Match descriptors using Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # return Nx1x2 arrays representing x and y coord
    return src_pts, dst_pts

def filter(mask, src_pts, dst_pts):
    '''Filter the keypoints based on mask'''
    # Nx1x2 arrays representing x and y coord
    src_pts = src_pts[mask.ravel()==1]
    dst_pts = dst_pts[mask.ravel()==1]

    return src_pts, dst_pts


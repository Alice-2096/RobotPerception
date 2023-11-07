
import cv2
import numpy as np

"""
For each captured image, we will compare it with the target images and see if there is a match.
@:param fpv: The captured image 
@:Prints the target image name if there is a match
"""

def detect(self, fpv):  
    def match_features(self, img1, img2):
        orb = cv2.ORB_create()

        # Find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # Create a brute force matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort them in ascending order of distance
        matches = sorted(matches, key=lambda x: x.distance)

        return len(matches) 

    targets = self.get_target_images()
    target_names = ["Front", "Left", "Back", "Right"]
    for index, target in enumerate(targets):
        match_score = self.match_features(fpv, target)
        # Check if the match score meets the threshold
        if match_score > 147:  # Using a threshold of 148
            print(f"Found target {target_names[index]}")
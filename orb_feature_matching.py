import cv2
import matplotlib.pyplot as plt

# Read images
img1 = cv2.imread('bottle.jpeg')
img2 = cv2.imread('bottle1.jpeg')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
keypoints_1, descriptors_1 = orb.detectAndCompute(gray1, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(gray2, None)

# Feature matching using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors_1, descriptors_2)

# Sort them in ascending order of distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the first 50 matches
img_matches = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:70], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
plt.imshow(img_matches)
plt.show()

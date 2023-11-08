import cv2
import numpy as np

print("Runnning Image Registration script...")
image_below = cv2.imread('../Images/B.jpg')
image_below = cv2.flip(image_below, 1)
image_below = cv2.rotate(image_below, cv2.ROTATE_180)

# Load the original color images
image_above = image_below 
image_below = cv2.imread('../Images/T.jpg')

# Initialize the feature detector and descriptor
detector = cv2.SIFT_create()
matcher = cv2.BFMatcher()

# Detect key points and compute descriptors
keypoints_above, descriptors_above = detector.detectAndCompute(image_above, None)
keypoints_below, descriptors_below = detector.detectAndCompute(image_below, None)

# Match descriptors
matches = matcher.knnMatch(descriptors_above, descriptors_below, k=2)

# Apply ratio test to select good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Sort matches by distance
good_matches = sorted(good_matches, key=lambda x: x.distance)

# Estimate the transformation matrix (homography) using RANSAC
src_pts = np.float32([keypoints_above[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints_below[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# Warp and align the original color image
aligned_color_image_above = cv2.warpPerspective(image_above, M, (image_below.shape[1], image_below.shape[0]))

# You can save the aligned color image if needed
cv2.imwrite('../Images/B.jpg', aligned_color_image_above)

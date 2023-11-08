import cv2
import numpy as np

print("Performing convex algorithm...")
image = cv2.imread('..\ResultMask.png', cv2.IMREAD_GRAYSCALE)

contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hull_image = np.zeros_like(image)

for contour in contours:
    hull = cv2.convexHull(contour)
    cv2.drawContours(hull_image, [hull], 0, 255, -1)

cv2.imwrite('../Convex.png', hull_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

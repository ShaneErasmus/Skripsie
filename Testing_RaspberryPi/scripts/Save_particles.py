import cv2
import numpy as np

def get_centers_and_contours(image, mask_image_path):
    final_result = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    image[final_result == 0] = [0, 0, 0]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raw_contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in raw_contours if cv2.contourArea(contour) > 2000]
    centers = [(x + w // 2, y + h // 2) for contour in contours for x, y, w, h in [cv2.boundingRect(contour)]]
    return centers, contours

def pad_image_to_size(image, target_width, target_height):
    y_padding = (target_height - image.shape[0]) // 2
    x_padding = (target_width - image.shape[1]) // 2
    return cv2.copyMakeBorder(image, y_padding, target_height - image.shape[0] - y_padding, x_padding, target_width - image.shape[1] - x_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

print("Saving Particles...")
path_base = '../Images'
kernels_T = cv2.imread(path_base + '/T.jpg')
kernels_B = cv2.imread(path_base + '/B.jpg')

centers_T, contours_T = get_centers_and_contours(kernels_T, '../FinalMaskT.png')
centers_B, contours_B = get_centers_and_contours(kernels_B, '../FinalMaskB.png')

matched_pairs = [(index_T, index_B) for index_T, center_T in enumerate(centers_T) for index_B, center_B in enumerate(centers_B) if np.linalg.norm(np.subtract(center_T, center_B)) <= 5]

successful_match_count = 0  # New counter to keep track of the saved pairs

for (index_T, index_B) in matched_pairs:
    images_to_save = []
    skip_pair = False
    for idx, kernel, contours, prefix in [(index_T, kernels_T, contours_T, "_T"), (index_B, kernels_B, contours_B, "_B")]:
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask = np.zeros_like(kernel)
        cv2.drawContours(mask, [contours[idx]], 0, (255, 255, 255), -1)
        cropped_image = cv2.bitwise_and(kernel[y:y+h, x:x+w], mask[y:y+h, x:x+w])
        if cropped_image.shape[0] > 125 or cropped_image.shape[1] > 125:
            skip_pair = True
            break
        padded_image = pad_image_to_size(cropped_image, 125, 125)
        images_to_save.append((f'../Images/Particles/{successful_match_count}{prefix}.jpg', padded_image))

    if not skip_pair:
        for path, image in images_to_save:
            cv2.imwrite(path, image)
        successful_match_count += 1  # Increment the successful match count only when the pair is saved

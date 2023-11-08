import numpy as np
import cv2
import tensorflow as tf

def get_centers_and_contours(image, mask_image_path):
    final_result = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    image[final_result == 0] = [0, 0, 0]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raw_contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in raw_contours if cv2.contourArea(contour) > 2200]
    centers = [(x + w // 2, y + h // 2) for contour in contours for x, y, w, h in [cv2.boundingRect(contour)]]
    return centers, contours

def pad_image_to_size(image, target_width, target_height):
    y_padding = (target_height - image.shape[0]) // 2
    x_padding = (target_width - image.shape[1]) // 2
    return cv2.copyMakeBorder(image, y_padding, target_height - image.shape[0] - y_padding, x_padding, target_width - image.shape[1] - x_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

print("Processing Particles...")
path_base = 'C:\\Users\\Shane\\Documents\\Universiteit\\Fourth Year\\Skripsie\\ALL_IMAGES\\SAGL_Images\\Grade_Images\\Batch3\\' #********************CHANGE TO PATH OF T and B image********************
kernels_T = cv2.imread(path_base + 'T.jpg')
kernels_B = cv2.imread(path_base + 'B.jpg')

centers_T, contours_T = get_centers_and_contours(kernels_T, '../FinalMaskT.png')
centers_B, contours_B = get_centers_and_contours(kernels_B, '../FinalMaskB.png')

matched_pairs = [(index_T, index_B) for index_T, center_T in enumerate(centers_T) for index_B, center_B in enumerate(centers_B) if np.linalg.norm(np.subtract(center_T, center_B)) <= 10]

save_count = 0 
images = []
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

        images_to_save.append(padded_image)
    
    if not skip_pair:      
        t_img = cv2.cvtColor(images_to_save[0], cv2.COLOR_BGR2RGB)
        b_img = cv2.cvtColor(images_to_save[1], cv2.COLOR_BGR2RGB)
        combined_img = np.concatenate([t_img,b_img], axis=2)
        #SAVE THE IMAGES
        #cv2.imwrite(f'Particles/{save_count}_T.jpg', t_img)
        #cv2.imwrite(f'Particles/{save_count}_B.jpg',b_img)
        images.append(combined_img)
        save_count += 1

print("-----------------------------")
print("Evaluating Particles...")
print("-----------------------------")
model = tf.keras.models.load_model('../THE_ONE.h5')
# Preprocess the images - normalize them
images = np.array(images)
images_normalized = images / 255.0

# Predict using the model
predictions = model.predict(images_normalized)

predicted_labels = np.argmax(predictions, axis=1)
class_descriptions = [
    "White Maize",
    "Discolored",
    "Yellow Maize",
    "Insect Damage",
    "Heat Damage",
    "Fusarium"
]

# Count occurrences of each class in the predicted labels
class_counts = np.bincount(predicted_labels, minlength=len(class_descriptions))
total_predictions = len(predicted_labels)

# Find the maximum length among all descriptions and static texts
max_desc_length = max(
    [len(desc) for desc in class_descriptions] +
    [len("Defects above 6.35mm"), len("Other colored"), len("Combined deviations")]
)

# Separators
separator = "-" * (max_desc_length + 19)  # 19 = len(":   100.00% (1000)") + 2 for spaces around |
start_separator = "-" * 3

print("\n\n\nReport:\n\n\n")
print(start_separator + separator)

Def_above = class_counts[3] + class_counts[4] + class_counts[5]
if class_counts[0] > class_counts[2]:
    Other_col = class_counts[1] + class_counts[2]
    Grade_class = 'WM'
else:
    Other_col = class_counts[1] + class_counts[0]
    Grade_class = 'YM'
Comb = Other_col + Def_above


#Determine Grade Def:
Def_per = Def_above/total_predictions*100
if 0 < Def_per < 7:
    grade_def = 1
elif 7 < Def_per < 13:
    grade_def = 2
elif 13 < Def_per < 30:
    grade_def = 3
else:
    grade_def = 100

#Determine Grade Col:
Other_col_per = Other_col/total_predictions*100
if 0 < Other_col_per < 3:
    grade_col = 1
elif 3 < Other_col_per < 6:
    grade_col = 2
elif 6 < Other_col_per < 10:
    grade_col = 3
else:
    grade_col = 100

#Determine Grade Comb:
Comb_per = Comb/total_predictions*100
if 0 < Other_col_per < 8:
    grade_com = 1
elif 8 < Other_col_per < 16:
    grade_com = 2
elif 16 < Other_col_per < 30:
    grade_com = 3
else:
    grade_com = 100


final_grade = max(grade_def,grade_col,grade_com)

max_line_width = len(start_separator + separator)

for desc, count in zip(class_descriptions, class_counts):
    percentage = (count / total_predictions) * 100
    line_content = f"| {desc:<{max_desc_length}}: {percentage:>7.2f}% ({count:4})"
    print(f"{line_content:<{max_line_width}}|")

print(start_separator + separator)

line_content = f"| {'Defects above 6.35mm':<{max_desc_length}}: {Def_above/total_predictions*100:>7.2f}% "
print(f"{line_content:<{max_line_width}}|")

line_content = f"| {'Other colored':<{max_desc_length}}: {Other_col/total_predictions*100:>7.2f}% "
print(f"{line_content:<{max_line_width}}|")

print(start_separator + separator)

line_content = f"| {'Combined deviations':<{max_desc_length}}: {Comb/total_predictions*100:>7.2f}% "
print(f"{line_content:<{max_line_width}}|")

print(start_separator + separator)

line_content = f"| {'Final Grade':<{max_desc_length}}:    {Grade_class + str(final_grade)}"
print(f"{line_content:<{max_line_width}}|")

print(start_separator + separator)
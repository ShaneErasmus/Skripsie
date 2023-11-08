import tensorflow as tf
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in range(6):  #CHANGE IF WANT ADD SF
        path = os.path.join(folder, str(label))
        for file in sorted(os.listdir(path)):
            if "_T.jpg" in file:
                t_img = cv2.imread(os.path.join(path, file))
                t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
                b_img_name = file.replace("_T.jpg", "_B.jpg")
                b_img = cv2.imread(os.path.join(path, b_img_name))
                b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
                combined_img = np.concatenate([t_img, b_img], axis=2)
                images.append(combined_img)
                labels.append(label)
    return np.array(images), np.array(labels)

images, labels = load_images_from_folder('Training_Data_Set')
np.set_printoptions(threshold=np.inf)
print(images.shape)

train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.15, random_state=42)
train_images = train_images / 255.0
val_images = val_images / 255.0

model = tf.keras.Sequential([
    Conv2D(16, kernel_size=(3,3), padding='same', activation='relu', input_shape=(125,125,6)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(6, activation='softmax')
])

model.summary()
optimizer = Adam(learning_rate=0.001) 
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(train_images)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
history = model.fit(datagen.flow(train_images, train_labels), epochs=70, validation_data=(val_images, val_labels), callbacks=[lr_reducer, early_stopping])

test_loss, val_acc = model.evaluate(val_images, val_labels, verbose=2)
print("\nValidation accuracy:", val_acc)

# Plotting training and validation accuracies
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Displaying the confusion matrix
y_pred = model.predict(val_images)
y_pred_classes = np.argmax(y_pred, axis=1)

confusion_mtx = confusion_matrix(val_labels, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

model.save("model.h5")



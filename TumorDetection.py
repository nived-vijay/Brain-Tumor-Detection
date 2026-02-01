import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

IMG_SIZE = (64, 64)
TRAIN_DIR = os.path.join("dataset", "Training")
TEST_DIR = os.path.join("dataset", "Testing")

def load_data(directory):
    images, labels = [], []
    class_names = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]
    class_map = {name: i for i, name in enumerate(class_names)}
    
    print(f"Loading data from {directory}...")
    for root, dirs, files in os.walk(directory):
        folder_name = os.path.basename(root)
        if folder_name in class_map:
            label = class_map[folder_name]
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(cv2.resize(img, IMG_SIZE))
                        labels.append(label)
    return np.array(images), np.array(labels), class_names

def main():
    X_train, y_train, class_names = load_data(TRAIN_DIR)
    
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]
    
    X_train_proc = X_train.reshape(-1, 64, 64, 1) / 255.0
    y_train_cat = to_categorical(y_train, num_classes=len(class_names))

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(class_names), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("--Model Training in process--")
    model.fit(X_train_proc, y_train_cat, epochs=10, batch_size=16, verbose=1, validation_split=0.1)

    test_images, test_labels, _ = load_data(TEST_DIR)
    idx = random.randint(0, len(test_images) - 1)
    img = test_images[idx]
    
    pred = model.predict(img.reshape(1, 64, 64, 1) / 255.0, verbose=0)[0]
    pred_idx = np.argmax(pred)
    type_name = class_names[pred_idx]
    
    if type_name == "No Tumor":
        status = "No Tumor Detected"
        display_type = "N/A"
    else:
        status = "Tumor Detected"
        display_type = type_name
    
    print(f"\nResult: {status}")
    if type_name != "No Tumor":
        print(f"Tumor Type: {display_type}")
    print(f"Confidence: {pred[pred_idx]*100:.2f}%")

    plt.imshow(img, cmap='gray')
    plt.title(f"Status: {status}\nType: {display_type}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()

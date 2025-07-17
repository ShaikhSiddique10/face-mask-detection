import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


data = []
labels = []


for category in ['with_mask', 'without_mask']:
    path = os.path.join('dataset', category)
    class_num = 0 if category == 'with_mask' else 1

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            img_array = cv2.resize(img_array, (100, 100))
            data.append(img_array)
            labels.append(class_num)
        except Exception as e:
            print(f"Error loading image {img}: {e}")


X = np.array(data) / 255.0
y = to_categorical(np.array(labels))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))


model.save('mask_detector.model')

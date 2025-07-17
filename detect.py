import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('mask_detector.model')
face_cascade = cv2.CascadeClassifier('haarcascade.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        resized = cv2.resize(face, (100, 100)) / 255.0
        reshaped = np.reshape(resized, (1, 100, 100, 3))

        result = model.predict(reshaped)
        label = np.argmax(result)
        color = (0, 255, 0) if label == 0 else (0, 0, 255)
        text = "Mask" if label == 0 else "No Mask"

        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Face Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

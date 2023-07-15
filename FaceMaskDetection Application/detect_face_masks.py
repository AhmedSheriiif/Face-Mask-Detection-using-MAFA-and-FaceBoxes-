import numpy as np
import cv2

# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import load_model

# Face Detector
from face_detect import FaceDetector

print("Loading Face Mask Detector Model...")
MaskDetectorMODEL = load_model('mobileNetv2_MAFA_model_mask_nomask_V2_Dense1.h5')
print("Face Mask Detector Model Loaded")
face_detector = FaceDetector()

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Frame width:', frame_width)
print('Frame height:', frame_height)
print('Capture frame rate:', cap.get(cv2.CAP_PROP_FPS))
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    faces, locations = face_detector.detect_face_in_frame(frame)

    # Detecting Face Masks
    predictions = []
    for i, face in enumerate(faces):
        # check if face's size is not empty
        try:
            if len(face) != 0:
                classes = ['full_mask', 'no mask']
                face_resized = cv2.resize(face, (160, 160))
                frame_arr = np.expand_dims(face_resized, axis=0)
                # Predictions
                pred = MaskDetectorMODEL.predict(frame_arr)
                predictions.append(pred)
                print(pred)

        except:
            pass

    if predictions is not None:  # if there is faces detected
        for pred, loc in zip(predictions, locations):
            (startX, startY, endX, endY) = loc

            # Checking Prediction   0 <--0.5-> 1
            pred = tf.nn.sigmoid(pred)
            pred = tf.where(pred < 0.5, 0, 1)
            pred = (pred.numpy()[0][0])
            if pred == 0:
                result = 'With Mask'
                frame = cv2.putText(frame, result, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 0), 1)
                frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 1)

            else:
                result = 'Without Mask'
                frame = cv2.putText(frame, result, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 1)
                frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

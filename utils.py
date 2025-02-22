import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras.models import load_model


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def get_face_crop(image):
    # BGR -> RGB

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_detection.process(image_rgb)

    if results.detections:

        
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box


        ih, iw, _ = image.shape


        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)

        if (x + w) - x > 0 and (y + h) - y > 0:

            face = image[y:y+h, x:x+w]
        
            return face
        return None
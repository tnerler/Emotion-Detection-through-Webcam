from utils import get_face_crop
from preprocess_frame import preprocess_frame
import cv2
from keras.models import load_model
import numpy as np 



model = load_model('best_model_1.h5')

emotions = ['ANGRY', 'DISGUSTED', 'FEARFUL', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISED']

webcam = cv2.VideoCapture(0)

ret, frame = webcam.read()

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    face_crop = get_face_crop(frame)
    
    
    if face_crop is None:
        cv2.imshow('frame', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
        continue

    preprocessed_face = preprocess_frame(face_crop)
    predictions = model.predict(preprocessed_face)
    emotion_idx = np.argmax(predictions[0])


    cv2.putText(frame,
                emotions[emotion_idx],
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                3)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

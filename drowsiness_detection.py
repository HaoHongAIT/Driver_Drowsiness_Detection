import os
import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer

mixer.init()
path = os.getcwd()

# face = cv2.CascadeClassifier('./haar_cascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('./haar_cascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('./haar_cascade_files/haarcascade_righteye_2splits.xml')
sound = mixer.Sound(r'sound_files/alarm.mp3')


class drowsiness:
    def __init__(self):
        self.model = load_model('./models/cnnCat2.h5')
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.score = 0
        self.rpred = [0]
        self.lpred = [0]

    def preprocess_eye(self, frame, x, y, w, h):
        lr_eye = frame[y:y + h, x:x + w]
        lr_eye = cv2.cvtColor(lr_eye, cv2.COLOR_BGR2GRAY)
        lr_eye = cv2.resize(lr_eye, (24, 24))
        lr_eye = lr_eye / 255
        lr_eye = lr_eye.reshape(24, 24, -1)
        lr_eye = np.expand_dims(lr_eye, axis=0)
        return lr_eye  # ảnh xám mắt trái/phải

    def detect(self, ret, frame):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)
        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        # faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for (x, y, w, h) in right_eye:
            r_eye = self.preprocess_eye(frame, x, y, w, h)
            self.rpred = np.argmax(self.model.predict(r_eye), axis=-1)
            break
        for (x, y, w, h) in left_eye:
            l_eye = self.preprocess_eye(frame, x, y, w, h)
            self.lpred = np.argmax(self.model.predict(l_eye), axis=-1)
            break

        # status = self.rpred[0] or self.lpred[0]
        # self.score = max(0, self.score + (-2 * status + 1))
        # lb = ["Closed", "Open"]
        # cv2.putText(frame, lb[status], (10, height - 20), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if self.rpred[0] == 0 and self.lpred[0] == 0:
            self.score = self.score + 1
            cv2.putText(frame, "Closed", (10, height - 20), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            self.score = max(0, self.score - 1)
            cv2.putText(frame, "Open", (10, height - 20), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if self.score > 15:
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            try:
                sound.play()
            except:  # isplaying = False
                pass

        cv2.putText(frame, 'Score:' + str(self.score), (100, height - 20), self.font, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        return cv2.imencode('.jpg', frame)[1].tobytes()  # frame

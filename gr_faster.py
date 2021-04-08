# The same genderrecognition.py code but with multi-threading to make it faster and fix the the lag of the other one
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
from threading import Thread

# open webcam and initiate the cam
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# opencv class
class VideoStream:
    def __init__(self):
        # read frame from webcam
        self.stream = cv2.VideoCapture(0)
        self.status, self.frame = self.stream.read()
        self.stopped= False
        # webcam.set(cv2.CAP_PROP_FPS, 1000)
        # self.frame = cv2.flip(self.frame, 1)
        print("videostream working")

    def start(self):
        Thread(target=self.stream_read, args=()).start()
        return self

    def stream_read(self):
        while not self.stopped:
            (self.status, self.frame) = self.stream.read()

    def stop_stream(self):
        self.stopped = True


# face detection class
class FaceDetection:
    def __init__(self, video_stream):
        # use VideoStream Class variables
        # self.videostream = VideoStream()
        # self.frame = self.videostream.frame

        # apply face detection

        self.stream = video_stream
        self.stopped = False
        self.output_frame = None

    def start(self):
        Thread(target=self.face_detection, args=()).start()  # Thread for face_detection
        return self

    def stop_process(self):
        self.stopped = True

    def face_detection(self):
        print("face_detection working")
        while not self.stopped:
            if self.stream is not None:
                frame = self.stream.frame  # This is where we get a frame from the video stream

                face, confidence = cv.detect_face(frame)
                # loop through detected faces
                for idx, f in enumerate(face):
                    # get the corner point of the rectangle
                    startX, startY = f[0], f[1]
                    endX, endY = f[2], f[3]

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    face_crop = np.copy(frame[startY:endY, startX:endX])

                    if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                        continue

                    # preprocessing for gender detection model
                    face_crop = cv2.resize(face_crop, (96,96))
                    face_crop = face_crop.astype("float") / 255.0
                    face_crop = img_to_array(face_crop)
                    face_crop = np.expand_dims(face_crop, axis=0)
                    self.output_frame = face_crop  # That will be the output frame to run through gender detection

            # GFR()   Taking this out because I can't get the model to work

# gender recognition class
class GFR:
    def __init__(self):
        self.model = load_model("C:/Users/berna/Desktop/Programming/AI_ML_DL/Projects/FaceGenderRecognition/gender_detection.model")
        self.facedetection = FaceDetection()

        self.face_crop = self.facedetection.face_crop
        self.classes = ['hombre', 'mujer']
        self.startX, self.startY = self.facedetection.startX, self.facedetection.startY
        self.endX, self.endY = self.facedetection.endX, self.facedetection.endY
        self.frame = self.facedetection.frame

        # apply the gender detection face with the model
        self.conf = model.predict(self.face_crop)[0]

        # get label with max acc
        self.idx = np.argmax(self.conf)
        self.label = self.classes[self.idx]

        self.label = "{}: {:.2f}".format(self.label, self.conf[self.idx] * 100)

        self.Y = self.startY - 10 if self.startY - 10 > 10 else self.startY + 10

        # write label and confidence above the face rectangle
        cv2.putText(self.frame, self.label, (self.startX, self.Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

        print("gender recognition working!")


# classes and webcam while loop
# gender_detection = GFR()  Taken out because I can't get it to work


# # loop through frames
# while webcam.isOpened():
#     VideoStream()
#     face_detection()
#
#     # display output
#     cv2.imshow("Gender Detection", gender_detection.frame)
#
#     # press "Q" to stop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


def face_recognition_display():
    video_stream = VideoStream().start()
    face_detection = FaceDetection(video_stream).start()

    while True:
        pressed_key = cv2.waitKey(1) & 0xFF

        if pressed_key == ord('q'):
            video_stream.stop_stream()
            face_detection.stop_process()
            print("Face detection stopped")
            cv2.destroyAllWindows()
            break

        frame = video_stream.frame  # Getting the most recent frame read by the video stream

        cv2.imshow("Output", frame)


# webcam.release()
if __name__ == '__main__':
    face_recognition_display()

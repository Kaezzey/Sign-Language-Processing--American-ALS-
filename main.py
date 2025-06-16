import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
import time

def mediapipe_detection(image, model):
    
    #colour conversion BGR to RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    #image is no longer writeable
    image.flags.writeable = False  

    #predict from the openCV frame using mediapipe and get results
    results = model.process(image)  

    #image is writeable again
    image.flags.writeable = True

    #convert the image color back to BGR
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    return image, results

def draw_format_landmarks(image, results):

    mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def main():
    
    #accessing webcam using OpenCV
    cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)

    #60fps camera feed
    cap.set(cv.CAP_PROP_FPS, 60)

    #set the mediapipe holistic model with specified confidence levels
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            
            #read frame from webcam
            ret, frame = cap.read()

            frame = cv.resize(frame, (720, 405))

            #make detection using mediapipe
            image, results = mediapipe_detection(frame, holistic)

            #draw landmarks on the image
            draw_format_landmarks(image, results)

            if not ret:

                break
            
            #flip horizontally
            image = cv.flip(image, 1)

            #show to screen
            cv.imshow('Webcam Feed', image)

            #press 'ESC' to exit
            if cv.waitKey(10) & 0xFF == 27:

                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    #using holistic model from mediapipe
    mp_holistic = mp.solutions.holistic

    #using drawing utilities from mediapipe
    mp_drawing = mp.solutions.drawing_utils

    main()
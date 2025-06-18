import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
import time
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from keras.callbacks import EarlyStopping

from keras.layers import Dropout

from keras.models import load_model

from sklearn.metrics import classification_report


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

    #draw landmarks on the image using mediapipe drawing utilities
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=2))

def extract_keypoints(results):

    #extract keypoints from the results

    #face has 468 landmarks, left hand has 21 landmarks, right hand has 21 landmarks, and pose has 33 landmarks
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

    return np.concatenate([pose, face, left_hand, right_hand])

def create_data_folders(actions, no_sequences):

    for action in actions:

        for sequence in range(no_sequences):

            os.makedirs(os.path.join(data_path, action, str(sequence)), exist_ok=True)

def collect_keypoints():

    #accessing webcam using OpenCV
    cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)

    #60fps camera feed
    cap.set(cv.CAP_PROP_FPS, 60)

    #set the mediapipe holistic model with specified confidence levels
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        #loop through the actions
        for action in actions:

            #loop through the videos
            for sequence in range(no_sequences):

                #loop through the frames of the videos
                for frame_num in range(sequence_length):
            
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

                    #wait logic
                    if frame_num == 0:

                        cv.putText(image, 'Collecting frames for action:', (120, 70),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv.LINE_AA)
                        cv.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                        
                        #show to screen
                        cv.imshow('Webcam Feed', image)
                        cv.waitKey(2000)

                    else:

                        cv.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                       
                        #show to screen
                        cv.imshow('Webcam Feed', image)

                    #extract keypoints from the results
                    keypoints= extract_keypoints(results)

                    #get path of the numpy file
                    npy_path = os.path.join(data_path, action, str(sequence), str(frame_num))

                    #save the keypoints to a numpy file at npy_path 
                    np.save(npy_path, keypoints)

                    #press 'ESC' to exit
                    if cv.waitKey(1) & 0xFF == 27:

                        break

    cap.release()
    cv.destroyAllWindows()

def create_label_dict(actions):

    #create a label dictionary for the actions
    label_dict = {action: num for num, action in enumerate(actions)}

    return label_dict

def load_data(label_dict):

    #load the data from the numpy files
    sequences, labels = [], []

    #loop through the actions
    for action in actions:

        #get the number of sequences for the action
        for sequence in range(no_sequences):

            #video
            window = []

            #loop through the frames of the video
            for frame_num in range(sequence_length):

                #get the path of the numpy file
                npy_path = os.path.join(data_path, action, str(sequence), '{}.npy'.format(frame_num))

                #load the numpy file
                if os.path.exists(npy_path):

                    #load the numpy file and append to the window
                    res = np.load(npy_path)
                    window.append(res)

            #append
            sequences.append(window)
            labels.append(label_dict[action])

    #convert to numpy arrays
    sequences, labels = np.array(sequences), to_categorical(np.array(labels)).astype(int)

    #split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2)

    return X_train, X_test, y_train, y_test

def model_training(label_dict, X_train, X_test, y_train, y_test):

    #logging the model training
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    #create the model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    #train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=256, callbacks=[tb_callback])

    model.summary()

    #save the model
    model.save('action_recognition_model.h5')
    print("Model saved successfully!")

    return model

def model_analysis(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print(classification_report(y_true_labels, y_pred_labels, target_names=actions))


def main(model):

    #sequence and sentence are used to store the keypoints and the actions respectively
    sequence = []
    sentence = []
    threshold = 0.4
    
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

            # flip horizontally
            image = cv.flip(image, 1)

            #prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            #keep the last 30 frames
            sequence = sequence[-30:]  

            if len(sequence) == 30:

                #make prediction using the model
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])

                #if the prediction is above the threshold, add it to the sentence
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0 and actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                    elif len(sentence) == 0:
                        sentence.append(actions[np.argmax(res)])

            # visualisation

                if res[np.argmax(res)] > threshold:

                    # display probabilities for each action on the left side
                    for idx, action in enumerate(actions):
                        prob = res[idx]
                        text = f"{action}: {prob:.2f}"
                        y = 100 + idx * 30
                        cv.putText(image, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if idx == np.argmax(res) else (255, 255, 255), 2, cv.LINE_AA)

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            # display the sentence on the image
            cv.rectangle(image, (0, 0), (720, 40), (0, 0, 0), -1)
            cv.putText(image, ' '.join(sentence), (3, 30), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv.LINE_AA)

            if not ret:

                break

            #show to screen
            cv.imshow('Webcam Feed', image)

            #press 'ESC' to exit
            if cv.waitKey(1) & 0xFF == 27:

                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":

    #keep these two for running the model after training
    #using holistic model from mediapipe
    mp_holistic = mp.solutions.holistic

    #using drawing utilities from mediapipe
    mp_drawing = mp.solutions.drawing_utils

    #comment this out if you want to run the model after training
    #set the path for saving the data
    data_path= os.path.join('mp_data')

    #actions to be recognized:
    #add more to the training later
    actions = np.array(['hello', 'thanks', 'iloveyou', 'yes', 'no'])

    #number of sequences to train
    no_sequences = 30

    #length of each sequence (5 actions * 30 vids * 30 frames = 4500 frames)
    sequence_length = 30

    create_data_folders(actions, no_sequences)

    label_dict = create_label_dict(actions)

    X_train, X_test, y_train, y_test = load_data(label_dict)

    #uncomment if you want to re-train the model
    #model = model_training(label_dict, X_train, X_test, y_train, y_test)

    model = load_model('action_recognition_model.h5')

    model.load_weights('action_recognition_model.h5')
    print("Model loaded successfully!")

    model_analysis(model, X_test, y_test)

    #main(model)
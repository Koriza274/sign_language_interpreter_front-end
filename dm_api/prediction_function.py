import cv2
import mediapipe as mp
import numpy as np

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import save_model, load_model

from matplotlib import pyplot as plt




def get_encoder():

    labels = np.load("labels.npy")



    # Encode labels as integers and convert to categorical
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    return label_encoder

def get_max_landmark_data():
    landmark_data =np.load("landmark_data.npy")
    return np.max(landmark_data)


def adjust_brightness_contrast(image, brightness=40, contrast=1.0):
    # Convert to float to prevent clipping
    img = image.astype(np.float32)
    # Adjust brightness and contrast
    img = img * contrast + brightness
    # Clip to keep pixel values between 0 and 255 and convert back to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def predict_image(directory):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,min_detection_confidence=0.4)
    mp_drawing = mp.solutions.drawing_utils
    model = load_model("/home/diana/code/Koriza274/sign_language_interpreter/asl_sign_language_model_tf_2.18.keras")

    img = cv2.imread(directory)
    img = adjust_brightness_contrast(img, 40, 1)

    img_rbg =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rbg)

    sequence = []
    sequence_length = 1

    label_encoder = get_encoder()

    if result.multi_hand_landmarks:
        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        # Draw hand landmarks on the frame
        mp_drawing.draw_landmarks(
            img,
            result.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS
        )

        # Append new frame landmarks to sequence
        sequence.append(landmarks)
        if len(sequence) > sequence_length:
            sequence.pop(0)

        if len(sequence) == sequence_length:
            sequence_input = np.array(sequence)
            sequence_input = sequence_input/ np.max(get_max_landmark_data())
            sequence_input = sequence_input.flatten()[np.newaxis, ..., np.newaxis]
            prediction = model.predict(sequence_input)
            predicted_label_index = np.argmax(prediction)
            predicted_label = label_encoder.inverse_transform([predicted_label_index])
            confidence = prediction[0][predicted_label_index]

    #plt.axis('off')
    #plt.imshow(img)


    return predicted_label[0]

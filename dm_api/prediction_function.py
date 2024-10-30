import cv2
import mediapipe as mp
import numpy as np

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import save_model, load_model

from matplotlib import pyplot as plt




def get_encoder():
    landmark_data =np.load("landmark_data.npy")
    labels = np.load("labels.npy")

    landmark_data = landmark_data / np.max(landmark_data)

    # Encode labels as integers and convert to categorical
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    return label_encoder




def predict_image(directory):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    model = load_model("/home/diana/code/Koriza274/sign_language_interpreter/asl_sign_language_model_tf_2.18.keras")

    img = cv2.imread(directory)

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
            sequence_input = np.array(sequence).flatten()[np.newaxis, ..., np.newaxis]
            prediction = model.predict(sequence_input)
            predicted_label_index = np.argmax(prediction)
            predicted_label = label_encoder.inverse_transform([predicted_label_index])
            confidence = prediction[0][predicted_label_index]

    plt.axis('off')
    plt.imshow(img)

    return predicted_label[0]

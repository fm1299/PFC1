import cv2
import numpy as np
import pywt
import tensorflow as tf
from tensorflow.python.keras import layers, models

# Function for Viola-Jones face detection


def detect_face(image_path):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # face_cascade = cv2.CascadeClassifier(
    #     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None

    # Assume only one face is present, take the first one
    x, y, w, h = faces[0]

    # Return the detected face region
    return gray[y:y+h, x:x+w]


# Image normalization and Histogram equalization
def normalize_image(face_image):
    # Normalize the image
    normalized_image = (face_image - np.min(face_image)) / \
        (np.max(face_image) - np.min(face_image))

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(
        (normalized_image * 255).astype(np.uint8))

    return equalized_image

# Function for stationary wavelet transform (SWT)

# def apply_swt(image):
#     coeffs = pywt.swt2(image, 'db1', level=1)
#     ll, (lh, hl, hh) = coeffs[0]
#     return ll, lh, hl, hh


def apply_swt(image):
    # Make the size even along both dimensions
    height, width = image.shape
    height = height - height % 2
    width = width - width % 2
    image = image[:height, :width]

    coeffs = pywt.swt2(image, 'db1', level=1)
    ll, (lh, hl, hh) = coeffs[0]
    return ll, lh, hl, hh


# Function for discrete cosine transform (DCT)


def apply_dct(subband):
    subband_dct = cv2.dct(np.float32(subband))
    return subband_dct[:8, :8]  # Take only the top-left 8x8 coefficients

# Function to create the neural network model


def create_neural_network(input_size, output_size):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='sigmoid', input_shape=(input_size,)))
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(output_size, activation='softmax'))

    return model

# Function to preprocess the dataset and convert labels to one-hot encoding


def preprocess_dataset(X, y, output_size):
    X_processed = []
    for image_path in X:
        face_image = detect_face(image_path)
        if face_image is not None:
            normalized_image = normalize_image(face_image)
            ll, lh, hl, hh = apply_swt(normalized_image)
            subbands = [lh, hl, hh]
            for subband in subbands:
                subband_dct = apply_dct(subband)
                X_processed.append(subband_dct.flatten())

    y_one_hot = tf.one_hot(y, depth=output_size)
    return np.array(X_processed), y_one_hot


# Load your dataset (replace these paths with your dataset paths)
# image_paths = ['path_to_image1.jpg', 'path_to_image2.jpg', ...]
image_paths = ['137.png']
# Assuming labels are integers corresponding to emotions
labels = [0, 1, 2, 3, 4, 5, 6]

# Preprocess the dataset
X_processed, y_one_hot = preprocess_dataset(image_paths, labels, output_size=7)

# Define the neural network
input_size = X_processed.shape[1]
output_size = 7  # Assuming 7 emotions
model = create_neural_network(input_size, output_size)

# Compile the model with the categorical cross-entropy loss and SGD optimizer
# optimize = tf.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_processed, y_one_hot, epochs=50,
          batch_size=32, validation_split=0.2)

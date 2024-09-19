import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Define paths
IMAGE_FOLDER = 'D:/HP Shared/Data Science/Paymate/qr_codes/images/train'  # Enter your image folder path here
LABEL_FOLDER = 'D:/HP Shared/Data Science/Paymate/qr_codes/labels/train'
IMAGE_SIZE = (128, 128)  # Resize the images to this size

# 1. Load the Dataset (images + labels)
def load_dataset(image_folder, label_folder):
    images = []
    labels = []
    
    # Loop through image files
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
            image_path = os.path.join(image_folder, file_name)
            
            # Load and preprocess the image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
            img = cv2.resize(img, IMAGE_SIZE)  # Resize image
            img = img / 255.0  # Normalize pixel values to [0, 1]
            images.append(img)

            # Load corresponding label from the .txt file
            label_file_name = os.path.splitext(file_name)[0] + '.txt'
            label_path = os.path.join(label_folder, label_file_name)
            
            with open(label_path, 'r') as label_file:
                label = label_file.read().strip()  # Read the label from the text file
                labels.append(label)
    
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension for CNN

    return images, labels

# 2. Preprocess the labels (encode them)
def preprocess_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels, label_encoder

# 3. Build the CNN Model
def build_qr_code_model(input_shape=(128, 128, 1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# 4. Main function to load data, train the model, and save it
def main():
    # Load the dataset
    images, labels = load_dataset(IMAGE_FOLDER, LABEL_FOLDER)
    
    # Preprocess labels
    encoded_labels, label_encoder = preprocess_labels(labels)
    
    # Split dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)
    
    # Build the model
    num_classes = len(np.unique(encoded_labels))  # Number of unique users
    model = build_qr_code_model(input_shape=(128, 128, 1), num_classes=num_classes)
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    
    # Save the trained model as .h5 file
    model.save('qr_code_model.h5')
    print("Model saved as qr_code_model.h5")
    
    # Save label encoder for future use
    np.save('label_encoder_classes.npy', label_encoder.classes_)  # Save label classes

if __name__ == '__main__':
    main()

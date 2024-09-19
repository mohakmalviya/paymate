import os
import base64
from pymongo import MongoClient
from PIL import Image
import io
from pymongo.server_api import ServerApi
import shutil
import tensorflow as tf
import numpy as np

# MongoDB setup
uri = os.getenv('MONGO_URI', 'mongodb+srv://sohamnsharma:rdcv4c75@payment.ujkvb.mongodb.net/?retryWrites=true&w=majority&appName=payment')
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['paymate']  # Database name

# Collections
user_collection = db['users']

# Directory to save the QR code images and labels
output_dir = "D:/HP Shared/Data Science/Paymate/qr_codes/images/train"
labels_dir = "D:/HP Shared/Data Science/Paymate/qr_codes/labels/train"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

def download_qr_codes():
    # Fetch users with QR codes from MongoDB
    users = user_collection.find({'qr_code': {'$exists': True}})

    # Iterate through the users and save each QR code as an image
    for user in users:
        username = user['username']
        qr_code_binary = user['qr_code']
        
        if qr_code_binary:
            # Convert the binary data to a base64 string and then back to an image
            qr_code_base64 = base64.b64encode(qr_code_binary).decode('utf-8')
            qr_code_data = base64.b64decode(qr_code_base64)
            
            # Create an image from the binary data
            qr_code_image = Image.open(io.BytesIO(qr_code_data))
            
            # Save the QR code image with a unique name
            image_path = os.path.join(output_dir, f"{username}_qr.png")
            qr_code_image.save(image_path)
            
            # Create an empty label file for the image
            label_path = os.path.join(labels_dir, f"{username}_qr.txt")
            with open(label_path, 'w') as label_file:
                # Assuming a single class and no annotations yet; adjust as needed
                label_file.write("0 0.5 0.5 1.0 1.0\n")  # Example entry
            
            print(f"Saved QR code for {username} at {image_path}")
            print(f"Created label file for {username} at {label_path}")

    print("QR code download complete.")

def parse_label_file(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    boxes = [list(map(float, line.strip().split())) for line in lines]
    return np.array(boxes)

def load_image_and_label(image_path, label_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0  # Normalize to [0, 1]

    labels = parse_label_file(label_path)
    return image, labels

def create_dataset(image_dir, label_dir, batch_size):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
    label_files = [os.path.join(label_dir, f.replace('.png', '.txt')) for f in image_files]

    dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))
    dataset = dataset.map(lambda img, lbl: tf.numpy_function(load_image_and_label, [img, lbl], [tf.float32, tf.float32]))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4)  # Assuming 4 coordinates for bounding boxes
    ])

    model.compile(optimizer='adam',
                  loss='mse',  # Mean squared error for bounding boxes
                  metrics=['accuracy'])
    return model

def train_model():
    dataset = create_dataset(output_dir, labels_dir, batch_size=16)

    model = create_model()
    model.fit(dataset, epochs=10)

    # Save the model
    model_save_path = 'D:/HP Shared/Data Science/Paymate/qr_codes/models/model.h5'
    model.save(model_save_path)
    print(f"Model saved as {model_save_path}")

    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_model_path = 'D:/HP Shared/Data Science/Paymate/qr_codes/models/model.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TensorFlow Lite model saved as {tflite_model_path}")

if __name__ == "__main__":
    # Step 1: Download QR codes
    download_qr_codes()

    # Step 2: Train TensorFlow model
    train_model()

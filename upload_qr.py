import cv2
import os

# Function to decode QR code from an image using OpenCV
def decode_qr_code(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Create a QRCodeDetector object
    detector = cv2.QRCodeDetector()
    
    # Detect and decode the QR code
    data, bbox, _ = detector.detectAndDecode(img)

    if data:
        return data
    else:
        print(f"No QR code found in {image_path}.")
        return None

# Path to the new directory
qr_code_dir = r'D:\HP Shared\Data Science\Paymate\qr_codes\images\train'

# List all files in the qr_codes directory
uploaded_files = os.listdir(qr_code_dir)

# Loop through uploaded files and decode QR codes
for filename in uploaded_files:
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Adjust extensions as needed
        file_path = os.path.join(qr_code_dir, filename)  # Get the full path
        print(f"Processing file: {file_path}")  # Print the full path
        decoded_text = decode_qr_code(file_path)  # Use the full path here
        if decoded_text:
            print(f"Decoded text: {decoded_text}")
        else:
            print("No QR code found.")

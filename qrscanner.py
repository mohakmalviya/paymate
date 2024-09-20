import cv2
from pyzbar import pyzbar

cap = cv2.VideoCapture(0)

# Variable to track if a QR code has been detected
detected = False

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    barcodes = pyzbar.decode(frame)
    
    for barcode in barcodes:
        if not detected:  # Only process if not already detected
            detected = True  # Set flag to True
            print(f"Type: {barcode.type}, Data: {barcode.data.decode('utf-8')}")
            break  # Exit the for loop after first detection

    if detected:
        break  # Exit the while loop if QR code has been detected

    cv2.imshow('QR Code Scanner', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

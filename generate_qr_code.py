import qrcode
from pymongo import MongoClient
from bson import Binary
from io import BytesIO
from pymongo.server_api import ServerApi
import os

# MongoDB setup
uri = os.getenv('MONGO_URI', 'mongodb+srv://sohamnsharma:rdcv4c75@payment.ujkvb.mongodb.net/?retryWrites=true&w=majority&appName=payment')
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['paymate']  # Database name

# Collections
user_collection = db['users']

def generate_upi_id(username):
    """Generate a UPI ID based on the username."""
    return f"{username}@paymate"

def generate_qr_code(data):
    """Generate a QR code for the provided data and return as binary."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    # Create an image for the QR code
    img = qr.make_image(fill='black', back_color='white')
    
    # Save the QR code image to a byte stream
    byte_stream = BytesIO()
    img.save(byte_stream, format='PNG')
    byte_stream.seek(0)
    
    # Return the binary data of the QR code image
    return Binary(byte_stream.getvalue())

def update_users_with_qr_code():
    """Generate UPI ID and QR code for users and update the database."""
    # Fetch the first 6 users
    users = user_collection.find().limit(6)

    for user in users:
        username = user['username']

        # Generate UPI ID
        upi_id = generate_upi_id(username)

        # Generate QR Code based on UPI ID
        qr_code_binary = generate_qr_code(upi_id)

        # Update the user document with the UPI ID and QR Code
        user_collection.update_one(
            {'_id': user['_id']},
            {'$set': {
                'upi_id': upi_id,
                'qr_code': qr_code_binary
            }}
        )

        print(f"Updated user {username} with UPI ID {upi_id} and QR code.")

if __name__ == '__main__':
    # Run the update function
    update_users_with_qr_code()

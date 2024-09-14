from faker import Faker
from pymongo import MongoClient
import random
from datetime import datetime, timedelta
from bson.objectid import ObjectId
from PIL import Image, ImageDraw
import io
import base64

# Initialize Faker instance
fake = Faker()

# Connect to MongoDB (Replace with your MongoDB connection details)
client = MongoClient('mongodb+srv://sohamnsharma:rdcv4c75@payment.ujkvb.mongodb.net/?retryWrites=true&w=majority&appName=payment')
db = client['paymate']  # Database name

# Collections
bank_collection = db['bank_data']
user_collection = db['users']
transaction_collection = db['transactions']

# Function to generate dummy images for KYC documents and return binary data
def generate_dummy_image(user_id, doc_type):
    img = Image.new('RGB', (400, 200), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    
    # Draw document type and user ID on the image
    d.text((10, 80), f"{doc_type} for User: {user_id}", fill=(255, 255, 255))
    
    # Convert image to binary (in-memory)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    
    # Return binary data (Base64 encoded)
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

# Function to generate random bank data with additional fields for 5 users
def generate_bank_data(num_records):
    bank_data = []
    users_data = []
    for _ in range(num_records):
        user_id = ObjectId()  # Unique user ID

        # Generate bank account number, bank name, and IFSC code
        account_number = fake.random_number(digits=12, fix_len=True)
        bank_name = fake.company()
        ifsc_code = fake.swift()[:11]  # Random IFSC-like code
        
        # Generate first name, last name, and nominee details
        first_name = fake.first_name()
        last_name = fake.last_name()
        nominee_name = fake.name()
        relationship = random.choice(['spouse', 'parent', 'sibling', 'friend', 'child'])

        # User data
        user_data = {
            "_id": user_id,
            "username": fake.user_name(),
            "password": fake.password(length=10),
            "email": fake.email(),
            "phone_number": fake.phone_number(),
            "first_name": first_name,
            "last_name": last_name,
            "nominee_name": nominee_name,
            "relationship": relationship,
            "address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state(),
                "zip_code": fake.zipcode()
            },
            "account_number": account_number,  # Bank account details in user data
            "bank_name": bank_name,
            "ifsc_code": ifsc_code,
            "created_at": fake.date_time_this_decade(),
            "kyc_complete": False,
            "payment_link": None,
            "qr_code_path": None
        }
        
        # Generate KYC documents
        kyc_documents = [
            {
                "document_type": "Address Proof",
                "document_binary": generate_dummy_image(user_id, "Address_Proof"),  # Store as binary data
                "upload_date": fake.date_time_this_year()
            },
            {
                "document_type": "ID Card",
                "document_binary": generate_dummy_image(user_id, "ID_Card"),  # Store as binary data
                "upload_date": fake.date_time_this_year()
            }
        ]
        
        # Bank data linked to the user
        bank_data_entry = {
            "_id": user_id,  # Match the bank account with the user via user_id
            "username": user_data["username"],
            "account_number": account_number,
            "bank_name": bank_name,
            "ifsc_code": ifsc_code,
            "balance": round(random.uniform(1000, 100000), 2),  # Balance between 1,000 to 100,000
            "kyc_approved": False,
            "created_at": user_data["created_at"],  # Same as user creation date
            "kyc_documents": kyc_documents  # Store KYC documents as binary data
        }
        
        bank_data.append(bank_data_entry)
        users_data.append(user_data)

    return bank_data, users_data

# Function to generate random transactions with transaction ID
def generate_transactions(num_records, users):
    transactions = []
    user_ids = [user["_id"] for user in users]  # Extract user IDs

    for _ in range(num_records):
        # Select random sender and receiver from the user list
        sender_id = random.choice(user_ids)
        receiver_id = random.choice(user_ids)
        
        # Ensure sender and receiver are different
        while receiver_id == sender_id:
            receiver_id = random.choice(user_ids)
        
        # Random transaction amount and date (within last 3 years)
        amount = round(random.uniform(100, 10000), 2)
        transaction_date = datetime.now() - timedelta(days=random.randint(1, 1095))  # Last 3 years
        
        transaction = {
            "_id": ObjectId(),
            "transaction_id": fake.uuid4(),  # Unique transaction ID
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "amount": amount,
            "transaction_date": transaction_date,
            "status": random.choice(['completed', 'failed', 'pending'])  # Random transaction status
        }
        
        transactions.append(transaction)

    return transactions

# Generate 5 records for bank data and users with KYC and other details
bank_data, users_data = generate_bank_data(5)

# Insert bank data and user data into MongoDB
bank_collection.insert_many(bank_data)
user_collection.insert_many(users_data)

print(f"Inserted {len(bank_data)} bank records and {len(users_data)} user records.")

# Generate 50 transaction records
transactions_data = generate_transactions(50, users_data)

# Insert transactions into MongoDB
transaction_collection.insert_many(transactions_data)

print(f"Inserted {len(transactions_data)} transaction records into MongoDB.")

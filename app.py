from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file, Response
from werkzeug.security import generate_password_hash, check_password_hash
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
import os
import qrcode
import io
from pymongo import MongoClient
from dotenv import load_dotenv
from pymongo.server_api import ServerApi
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import string
from datetime import datetime
from bson.objectid import ObjectId
from functools import wraps
from flask import redirect, url_for, session
from pymongo import MongoClient
from gridfs import GridFS
import pandas as pd
from bson.decimal128 import Decimal128
from io import BytesIO
from pyzbar.pyzbar import decode
from PIL import Image
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model # type: ignore
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)
import yaml



load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback_secret_key')

# MongoDB setup
uri = os.getenv('MONGO_URI', 'mongodb+srv://sohamnsharma:rdcv4c75@payment.ujkvb.mongodb.net/?retryWrites=true&w=majority&appName=payment')
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['paymate']  # Database name

# Collections
bank_collection = db['bank_data']
user_collection = db['users']
transaction_collection = db['transactions']

# Email configuration
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')

# Encryption functions
fs = GridFS(db)

def save_kyc_documents(user_id, id_proof, address_proof):
    # Save ID proof
    id_proof_file = fs.put(id_proof, filename=f'{user_id}_id_proof')
    
    # Save Address proof
    address_proof_file = fs.put(address_proof, filename=f'{user_id}_address_proof')

    # Update user document with references to KYC documents
    user_collection.update_one(
        {'_id': user_id},
        {'$set': {
            'id_proof': id_proof_file,
            'address_proof': address_proof_file,
            'kyc_status': 'submitted'
        }}
    )

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def generate_key():
    return base64.urlsafe_b64encode(os.urandom(32))

def get_current_user():
    if 'username' not in session:
        return None
    return user_collection.find_one({'username': session['username']})

def encrypt_data(key, data):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(data.encode(), AES.block_size))
    iv = base64.b64encode(cipher.iv).decode('utf-8')
    ct = base64.b64encode(ct_bytes).decode('utf-8')
    return iv, ct

def decrypt_data(key, iv, ct):
    iv = base64.b64decode(iv)
    ct = base64.b64decode(ct)
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size).decode('utf-8')
    return pt


# Email verification functions
def generate_verification_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def send_verification_email(to_email, verification_code):
    message = MIMEMultipart("alternative")
    message["Subject"] = "Email Verification"
    message["From"] = SMTP_USERNAME
    message["To"] = to_email

    text = f"Your verification code is: {verification_code}"
    html = f"""\
    <html>
      <body>
        <p>Your verification code is: <strong>{verification_code}</strong></p>
      </body>
    </html>
    """

    message.attach(MIMEText(text, "plain"))
    message.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, to_email, message.as_string())
    except smtplib.SMTPAuthenticationError:
        print("Failed to authenticate with the SMTP server.")
    except smtplib.SMTPConnectError:
        print("Failed to connect to the SMTP server.")
    except smtplib.SMTPException as e:
        print(f"SMTP error occurred: {e}")
    except Exception as e:
        print(f"Error sending email: {e}")

#Update Functions
def get_user_bank_details(user_id):
    user = user_collection.find_one({'_id': ObjectId(user_id)})
    bank_account_id = user.get('bank_account_id')
    if bank_account_id:
        return bank_collection.find_one({'_id': ObjectId(bank_account_id)})
    return None

def get_last_10_transactions(user_id):
    user = user_collection.find_one({'_id': ObjectId(user_id)})
    username = user['username']
    return list(transaction_collection.find({
        '$or': [
            {'sender': username},
            {'recipient': username}
        ]
    }).sort('timestamp', -1).limit(10))

def update_user_username(user_id, new_username):
    user_collection.update_one(
        {'_id': ObjectId(user_id)},
        {'$set': {'username': new_username}}
    )

def update_user_account_number(user_id, new_account_number):
    user = user_collection.find_one({'_id': ObjectId(user_id)})
    bank_account_id = user.get('bank_account_id')
    if bank_account_id:
        bank_collection.update_one(
            {'_id': ObjectId(bank_account_id)},
            {'$set': {'account_number': new_account_number}}
        )

def update_user_bank_name(user_id, new_bank_name):
    user = user_collection.find_one({'_id': ObjectId(user_id)})
    bank_account_id = user.get('bank_account_id')
    if bank_account_id:
        bank_collection.update_one(
            {'_id': ObjectId(bank_account_id)},
            {'$set': {'bank_name': new_bank_name}}
        )

def update_user_ifsc_code(user_id, new_ifsc_code):
    user = user_collection.find_one({'_id': ObjectId(user_id)})
    bank_account_id = user.get('bank_account_id')
    if bank_account_id:
        bank_collection.update_one(
            {'_id': ObjectId(bank_account_id)},
            {'$set': {'ifsc_code': new_ifsc_code}}
        )

def update_user_balance(user_id, new_balance):
    user = user_collection.find_one({'_id': ObjectId(user_id)})
    bank_account_id = user.get('bank_account_id')
    if bank_account_id:
        bank_collection.update_one(
            {'_id': ObjectId(bank_account_id)},
            {'$set': {'balance': float(new_balance)}}
        )

def current_user():
    if 'username' not in session:
        return None
    return user_collection.find_one({'username': session['username']})

#Decode Amount Function
def decode_amount(encoded_amount):
    try:
        # Decode from base64
        decoded_bytes = base64.b64decode(encoded_amount)
        # Convert bytes to string and then to float
        decoded_amount = decoded_bytes.decode('utf-8')
        return "{:.2f}".format(float(decoded_amount))
    except Exception as e:
        # Handle decoding errors
        print(f"Error decoding amount: {e}")
        return '0.00'

# Routes
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        if 'verification_code' not in session:
            # First submission: store user data and send verification code
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')

            if not all([username, email, password]):
                flash('All fields are required.')
                return render_template('signup.html')

            existing_user = user_collection.find_one({'$or': [{'username': username}, {'email': email}]})
            if existing_user:
                flash('Username or email already exists.')
                return render_template('signup.html')

            session['temp_user_data'] = {
                'username': username,
                'email': email,
                'password': password
            }
            verification_code = generate_verification_code()
            session['verification_code'] = verification_code
            send_verification_email(email, verification_code)
            return render_template('signup.html', email_sent=True)
        else:
            # Verification code submission
            verification_code = request.form.get('verification_code')
            if verification_code == session['verification_code']:
                # Create user account
                user_data = session['temp_user_data']
                key = os.urandom(32)
                iv, encrypted_password = encrypt_data(key, user_data['password'])
                encoded_key = base64.b64encode(key).decode('utf-8')

                user_collection.insert_one({
                    'username': user_data['username'],
                    'email': user_data['email'],
                    'password': encrypted_password,
                    'iv': iv,
                    'key': encoded_key,
                    'kyc_complete': False,
                    'email_verified': True,
                    'payment_link': None,
                    'qr_code_path': None,
                    'is_admin': False,
                    'bank_account_id': None  # Default value
                })

                session.pop('verification_code', None)
                session.pop('temp_user_data', None)
                flash('Signup successful! Please log in.')
                return redirect(url_for('login'))
            else:
                flash('Invalid verification code. Please try again.')
                return render_template('signup.html', email_sent=True)

    return render_template('signup.html')

@app.route('/resend-verification', methods=['POST'])
def resend_verification():
    if 'temp_user_data' not in session:
        return jsonify({'success': False, 'message': 'No pending registration found.'})

    email = session['temp_user_data']['email']
    verification_code = generate_verification_code()
    session['verification_code'] = verification_code
    send_verification_email(email, verification_code)

    return jsonify({'success': True, 'message': 'Verification code resent.'})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = user_collection.find_one({'username': username})
        if user:
            key = base64.b64decode(user['key'])
            decrypted_password = decrypt_data(key, user['iv'], user['password'])
            
            if decrypted_password == password:
                session['username'] = username
                flash('Login successful!')
                
                # Redirect to onboarding if user hasn't completed it
                if user.get('kyc_complete') is False:
                    return redirect(url_for('onboarding'))
                return redirect(url_for('dashboard'))
            else:
                flash('Incorrect password. Please try again.')
        else:
            flash('Username does not exist.')
    
    return render_template('login.html')

@app.route('/onboarding', methods=['GET', 'POST'])
def onboarding():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    user = user_collection.find_one({'username': session['username']})
    
    if request.method == 'POST':
        # Handle document submission
        address_proof = request.files['address_proof']
        id_card = request.files['id_card']
        
        # Save documents
        address_proof_path = f"static/kyc/{session['username']}_address_proof.png"
        id_card_path = f"static/kyc/{session['username']}_id_card.png"
        address_proof.save(address_proof_path)
        id_card.save(id_card_path)
        
        # Update user document
        user_collection.update_one(
            {'username': session['username']},
            {'$set': {
                'kyc_complete': False,
                'address_proof': address_proof_path,
                'id_card': id_card_path,
                'kyc_status': 'pending'
            }}
        )
        
        flash('Documents submitted successfully. Waiting for admin approval.')
        return redirect(url_for('dashboard'))
    
    return render_template('onboarding.html', user=user)
    
    user = user_collection.find_one({'username': session['username']})
    return render_template('onboarding.html', user=user)


# Admin route with pagination
@app.route('/admin', methods=['GET'])
def admin():
    # Ensure only logged-in admin users can access this page
    if 'username' not in session or not user_collection.find_one({'username': session['username'], 'is_admin': True}):
        return redirect(url_for('login'))
    
    # Retrieve the logged-in admin user
    user = user_collection.find_one({'username': session['username']})

    # Pagination parameters
    page = int(request.args.get('page', 1))
    per_page = 500
    skip = (page - 1) * per_page

    # Fetch user data with pagination, including QR codes
    users = list(user_collection.find({}, {'username': 1, 'email': 1, 'kyc_status': 1, 'bank_balance': 1, 'qr_code': 1}).skip(skip).limit(per_page))

    # Convert QR code to base64 and handle Decimal128 balance fields
    for user in users:
        if isinstance(user.get('bank_balance'), Decimal128):
            user['bank_balance'] = str(user['bank_balance'])
        if user.get('qr_code'):
            # Convert binary QR code to base64 string for display
            qr_code_binary = user.get('qr_code')
            user['qr_code_base64'] = base64.b64encode(qr_code_binary).decode('utf-8')
        else:
            user['qr_code_base64'] = None  # If no QR code available

    # Fetch bank data and transactions with pagination
    bank_data = list(bank_collection.find().skip(skip).limit(per_page))
    transactions = list(transaction_collection.find().skip(skip).limit(per_page))

    # Calculate total pages for pagination
    total_banks = bank_collection.count_documents({})
    total_users = user_collection.count_documents({})
    total_transactions = transaction_collection.count_documents({})
    total_pages = max(total_banks, total_users, total_transactions) // per_page + 1

    # Page numbers for pagination
    pages = list(range(1, total_pages + 1))
    prev_page = max(page - 1, 1)
    next_page = min(page + 1, total_pages)

    # Fetch pending KYC approvals
    pending_kyc = user_collection.find({'kyc_status': 'pending'})

    # Render the admin template with user data and QR codes
    return render_template('admin.html',
                           user=user,
                           pending_kyc=pending_kyc,
                           bank_data=bank_data,
                           users=users,
                           transactions=transactions,
                           pages=pages,
                           current_page=page,
                           prev_page=prev_page,
                           next_page=next_page)

@app.route('/claim_onboarding_amount', methods=['POST'])
def claim_onboarding_amount():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    user = user_collection.find_one({'username': session['username']})
    
    if not user.get('kyc_complete'):
        flash('KYC not approved yet. Please complete KYC process first.')
        return redirect(url_for('onboarding'))
    
    if user.get('balance') is None:
        # Initialize balance to 10,000
        bank_account_id = user.get('bank_account_id')
        if bank_account_id:
            bank_collection.update_one(
                {'_id': bank_account_id},
                {'$set': {'balance': 10000}}
            )
            user_collection.update_one(
                {'username': session['username']},
                {'$set': {'balance': 10000}}
            )
            flash('Onboarding amount of 10,000 credited to your account.')
        else:
            flash('Bank account details are missing.')
    
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    user = user_collection.find_one({'username': session['username']})
    
    # Retrieve bank account details
    bank_account_id = user.get('bank_account_id')
    bank_account = {}
    if bank_account_id:
        bank_account = bank_collection.find_one({'_id': ObjectId(bank_account_id)})
    
    transactions = transaction_collection.find({
        '$or': [
            {'sender': session['username']},
            {'recipient': session['username']}
        ]
    }).sort('timestamp', -1).limit(10)
    
    return render_template('dashboard.html', user=user, bank_account=bank_account, transactions=transactions)


@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Fetch user data
    user = user_collection.find_one({'username': session['username']})

    if not user:
        return redirect(url_for('login'))  # Redirect if user is not found

    # Fetch user bank details
    bank_account_id = user.get('bank_account_id')
    bank = None
    if bank_account_id:
        bank = bank_collection.find_one({'_id': ObjectId(bank_account_id)})

    # Format bank balance to two decimal places
    if bank and 'balance' in bank:
        bank['balance'] = "{:.2f}".format(bank['balance'])

    # Fetch the last 3 transactions involving this user
    transactions = list(transaction_collection.find({
        '$or': [
            {'sender': session['username']},
            {'recipient': session['username']}
        ]
    }).sort('timestamp', -1).limit(3))

    # Add formatted transaction_date to each transaction
    for transaction in transactions:
        if 'timestamp' in transaction:
            transaction['transaction_date'] = transaction['timestamp'].strftime('%Y-%m-%d')

    # Create Paymate link
    paymate_link = f"https://paymate.com/{user['username']}"

    # Retrieve QR code as a base64 string
    qr_code_binary = user.get('qr_code')
    qr_code_base64 = base64.b64encode(qr_code_binary).decode('utf-8') if qr_code_binary else None

    return render_template('profile.html', user=user, bank=bank, transactions=transactions, paymate_link=paymate_link, qr_code=qr_code_base64)

@app.route('/update_kyc', methods=['POST'])
@login_required
def update_kyc():
    user = get_current_user()
    
    id_proof = request.files['id_proof']
    address_proof = request.files['address_proof']
    
    # Store KYC files in MongoDB (or GridFS) and update user's KYC status
    save_kyc_documents(user['_id'], id_proof, address_proof)
    
    flash("KYC documents uploaded successfully.")
    return redirect(url_for('profile'))

# Load the pre-trained model and label encoder
model = load_model('D:\HP Shared\Data Science\Paymate\qr_code_model.h5')

# Load the LabelEncoder from a .npy file
label_encoder = np.load('D:\HP Shared\Data Science\Paymate\label_encoder_classes.npy', allow_pickle=True).item()

model_input_size = 128  # Adjust according to your model's input size

def preprocess_image(image):
    resized_image = cv2.resize(image, (model_input_size, model_input_size))  # Adjust size based on model
    normalized_image = resized_image / 255.0  # Normalize if necessary
    return np.expand_dims(normalized_image, axis=0)

def decode_qr_code(qr_code_image):
    # Convert the image to a format suitable for your model
    image = cv2.imdecode(np.frombuffer(qr_code_image.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None

    image_array = preprocess_image(image)
    predictions = model.predict(image_array)
    predicted_class_idx = np.argmax(predictions)
    detected_username = label_encoder.inverse_transform([predicted_class_idx])[0]
    return detected_username

def generate():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        qr_codes = decode(frame)
        detected_username = None

        for qr_code in qr_codes:
            qr_data = qr_code.data.decode('utf-8')
            # For demonstration, assume qr_data is a path to an image
            # You might need to handle this part based on your QR code content
            image = cv2.imdecode(np.frombuffer(qr_data.encode(), np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                image_array = preprocess_image(image)
                predictions = model.predict(image_array)
                predicted_class_idx = np.argmax(predictions)
                detected_username = label_encoder.inverse_transform([predicted_class_idx])[0]

                points = qr_code.polygon
                if len(points) == 4:
                    pts = np.array([(point.x, point.y) for point in points], dtype=np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, detected_username, (qr_code.rect.left, qr_code.rect.top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

#Yaml config loader
import yaml

def load_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Example of how you might use the config for device checks:
def is_mobile_device(user_agent):
    if config['device_detection']['enabled']:
        allowed_platforms = config['device_detection']['platforms_allowed']
        return any(platform in user_agent.lower() for platform in ['mobi', 'android', 'iphone', 'ipad'])
    return False

@app.route('/scan_qr')
def scan_qr():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/transaction', methods=['GET', 'POST'])
def transaction():
    if 'username' not in session:
        return redirect(url_for('login'))

    user = user_collection.find_one({'username': session['username']})

    if isinstance(user.get('bank_balance'), Decimal128):
        bank_balance = float(user['bank_balance'].to_decimal())
    else:
        bank_balance = 0.0

    formatted_balance = f"{bank_balance:.2f}"

    qr_code_binary = user.get('qr_code')
    qr_code_base64 = base64.b64encode(qr_code_binary).decode('utf-8') if qr_code_binary else None

    if request.method == 'POST':
        method = request.form.get('method')
        recipient = request.form.get('recipient')
        try:
            amount = float(request.form.get('amount', 0))
        except ValueError:
            flash('Invalid amount.')
            return redirect(url_for('transaction'))

        sender_user = user_collection.find_one({'username': session['username']})
        recipient_user = None

        if method == 'payment_link':
            recipient_user = user_collection.find_one({'payment_link': recipient})
        elif method == 'upi_id':
            recipient_user = user_collection.find_one({'upi_id': recipient})
        elif method == 'qr_code':
            if 'qr_code_image' in request.files:
                qr_code_image = request.files['qr_code_image']
                if qr_code_image:
                    detected_username = decode_qr_code(qr_code_image)
                    if detected_username:
                        recipient_user = user_collection.find_one({'username': detected_username})
                        if not recipient_user:
                            flash('Invalid QR code. User not found.')
                            return redirect(url_for('transaction'))
                    else:
                        flash('No QR code detected in the image.')
                        return redirect(url_for('transaction'))
                else:
                    flash('No image uploaded.')
                    return redirect(url_for('transaction'))
            else:
                flash('No image file provided.')
                return redirect(url_for('transaction'))

        if recipient_user:
            if amount <= bank_balance:
                user_collection.update_one(
                    {'username': session['username']},
                    {'$inc': {'bank_balance': -amount}}
                )
                
                user_collection.update_one(
                    {'username': recipient_user['username']},
                    {'$inc': {'bank_balance': amount}}
                )

                key = base64.b64decode(sender_user['key'])
                encrypted_amount, iv = encrypt_data(key, str(amount))
                
                transaction_collection.insert_one({
                    'sender': session['username'],
                    'recipient': recipient_user['username'],
                    'amount': encrypted_amount,
                    'iv': iv,
                    'timestamp': datetime.utcnow()
                })
                
                flash('Transaction successful!')
                return redirect(url_for('dashboard'))
            else:
                flash('Insufficient balance.')
        else:
            flash('Invalid recipient.')

    return render_template('transaction.html', user=user, bank_balance=formatted_balance, qr_code=qr_code_base64)

@app.route('/upload_qr', methods=['POST'])
def upload_qr():
    if 'username' not in session:
        return redirect(url_for('login'))

    user = user_collection.find_one({'username': session['username']})

    if 'qr_code_image' in request.files:
        qr_code_image = request.files['qr_code_image']
        if qr_code_image:
            detected_username = decode_qr_code(qr_code_image)
            if detected_username:
                recipient_user = user_collection.find_one({'username': detected_username})
                if recipient_user:
                    return render_template('transaction.html', user=user, qr_code=base64.b64encode(user.get('qr_code')).decode('utf-8'), recipient_user=recipient_user)
                else:
                    flash('No user found with the QR code.')
            else:
                flash('No QR code detected in the image.')
        else:
            flash('No image uploaded.')
    else:
        flash('No image file provided.')
    
    return redirect(url_for('transaction'))

@app.route('/bank-details')
@login_required  # Ensure user is logged in
def bank_details():
    # Fetch the currently logged-in user using session data
    username = session.get('username')
    if not username:
        flash('Please log in to view your bank details.', 'error')
        return redirect(url_for('login'))

    # Fetch user details from the users collection
    user = user_collection.find_one({'username': username})
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    # Fetch the bank details if the user has a bank_account_id
    bank = None
    if 'bank_account_id' in user:
        bank = bank_collection.find_one({'_id': ObjectId(user['bank_account_id'])})

    # Fetch the last 10 transactions
    transactions = get_last_10_transactions(user['_id'])

    return render_template(
        'bank_details.html',
        user=user,
        bank=bank,
        transactions=transactions
    )

@app.route('/update-username', methods=['POST'])
@login_required
def update_username():
    current_user = get_current_user()
    if not current_user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    new_username = request.form.get('new_username')
    if new_username:
        update_user_username(current_user['_id'], new_username)
        flash('Username updated successfully!', 'success')
    else:
        flash('Username cannot be empty.', 'error')
    return redirect(url_for('bank_details'))

@app.route('/update-bank-details', methods=['POST'])
@login_required
def update_bank_details():
    current_user = get_current_user()
    if not current_user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    new_account_number = request.form.get('new_account_number')
    new_bank_name = request.form.get('new_bank_name')
    new_ifsc_code = request.form.get('new_ifsc_code')
    new_balance = request.form.get('new_balance')

    if new_account_number: update_user_account_number(current_user['_id'], new_account_number)
    if new_bank_name: update_user_bank_name(current_user['_id'], new_bank_name)
    if new_ifsc_code: update_user_ifsc_code(current_user['_id'], new_ifsc_code)
    if new_balance: update_user_balance(current_user['_id'], new_balance)

    flash('Bank details updated successfully!', 'success')
    return redirect(url_for('bank_details'))

@app.route('/download-transactions')
@login_required
def download_transactions():
    current_user = get_current_user()
    if not current_user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    transactions = get_last_10_transactions(current_user['_id'])

    # Convert transactions to a DataFrame and then to CSV
    df = pd.DataFrame([{
        'Date': t['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),  # Format the datetime
        'Amount': decrypt_data(base64.b64decode(current_user['key']), t['iv'], t['amount']),
        'Sender': t['sender'],
        'Recipient': t['recipient']
    } for t in transactions])

    csv = df.to_csv(index=False)
    return send_file(
        io.StringIO(csv),
        mimetype='text/csv',
        as_attachment=True,
        download_name='transactions.csv'
    )

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('admin', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    #port = int(os.environ.get("PORT", 8000))  # Default to port 8000 if PORT is not set
    app.run(debug=True)
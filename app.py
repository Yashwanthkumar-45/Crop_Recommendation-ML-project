

import os
from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_migrate import Migrate  # Import Flask-Migrate

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your actual secret key

# Configure the SQLAlchemy part of the app
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # Use SQLite DB file 'site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Needed for sessions and forms

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirect unauthorized users to login page
migrate = Migrate(app, db)  # Initialize Flask-Migrate

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Ensure that 'current_user' is passed to all templates automatically
@app.context_processor
def inject_user():
    return dict(current_user=current_user)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False)
    review_text = db.Column(db.String(500), nullable=False)
    rating = db.Column(db.Integer, nullable=False)  # 1 to 5 scale

    def __repr__(self):
        return f'<Review {self.username}>'

# Load the trained model
try:
    with open('crop_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model file 'crop_model.pkl' not found. Please ensure it is in the correct directory.")

# Helper function to find crop image with any extension
def find_crop_image(predicted_crop):
    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'webp']  # Supported image extensions
    for ext in image_extensions:
        image_path = f"static/crop_images/{predicted_crop.lower()}.{ext}"
        if os.path.exists(image_path):
            return f"crop_images/{predicted_crop.lower()}.{ext}"
    return None  # Return None if no image is found

# Route for the home page
@app.route('/')
def home():
    reviews = Review.query.all()  # Fetch all reviews from the database
    return render_template('home.html', reviews=reviews)

# Route for the index page (form submission)
@app.route('/index')
@login_required  # Require login to access the crop prediction system
def index():
    return render_template('index.html')

# Route for handling form submissions and predicting the crop
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Get the form data (soil and weather conditions)
            N = float(request.form['nitrogen'])
            P = float(request.form['phosphorus'])
            K = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            pH = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            
            # Create a numpy array for model prediction
            data = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
            
            # Predict the crop
            prediction = model.predict(data)
            predicted_crop = prediction[0] if len(prediction) > 0 else "Unknown"
            
            # Find the crop image with any file extension
            crop_image = find_crop_image(predicted_crop)
            
            return render_template('index.html', prediction=predicted_crop, crop_image=crop_image)
    except ValueError:
        return render_template('index.html', prediction='Please enter valid numbers for all inputs.', crop_image=None)
    except Exception as e:
        return render_template('index.html', prediction=f'An error occurred: {str(e)}', crop_image=None)

# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Add user to the database
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))
    return render_template('register.html')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))  # Redirect to the main app after login
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

# Route for user logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Dashboard route for logged in users
@app.route('/dashboard')
@login_required
def dashboard():
    return redirect(url_for('home'))  # Redirect to index.html or another template

@app.route('/submit_review', methods=['POST'])
@login_required
def submit_review():
    if request.method == 'POST':
        review_text = request.form.get('review')
        rating = request.form.get('rating')
        username = current_user.username  # Get the current user's username
        
        new_review = Review(username=username, review_text=review_text, rating=rating)  # Include rating

        db.session.add(new_review)
        db.session.commit()
        
        return redirect(url_for('home'))



if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    app.run(debug=True)

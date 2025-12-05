import os
import uuid
import io
import base64
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join("static", "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    analyses = db.relationship("CropData", backref="user", lazy=True)


class CropData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    crop_type = db.Column(db.String(50), nullable=False)
    disease_status = db.Column(db.Integer, nullable=False)  
    disease_name = db.Column(db.String(100), nullable=True)
    yield_prediction = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
disease_models = {}
disease_classes = {}

CROP_CONFIGS = {
    "Millet": {
        "model_path": "models/millet_disease_model.pth",
        "class_names": ['Healthy', 'blast', 'rust']
    },
    "Sugarcane": {
        "model_path": "models/sugarcane_disease_model.pth",
        "class_names": ['BacterialBlights', 'Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
    },
    "Wheat": {
        "model_path": "models/wheat_disease_model.pth",
        "class_names": ['BlackPoint', 'FusariumFootRot', 'HealthyLeaf', 'LeafBlight', 'WheatBlast']
    },
    "Rice": {
        "model_path": "models/rice_disease_model.pth",
        "class_names": ['Bacterialblight', 'Brownspot', 'Leafsmut']
    },
    "Onion": {
        "model_path": "models/onion_disease_model.pth",
        "class_names": ['Iris yellow virus', 'Stemphylium leaf blight and collectrichum leaf blight', 'healthy', 'purple blotch']
    },
    "Maize": {   
        "model_path": "models/maize_disease_model.pth",
        "class_names": ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    },
    "Cotton(lint)": {   
        "model_path": "models/cotton_disease_model.pth",
        "class_names": ['bacterial_blight', 'curl_virus', 'fussarium_wilt', 'healthy']
    },
    "Soyabean": {   
        "model_path": "models/soyabean_disease_model.pth",
        "class_names": ['Caterpillar', 'Diabrotica speciosa', 'Healthy']
    }
}


def load_model(model_path, num_classes):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

for crop, cfg in CROP_CONFIGS.items():
    try:
        model = load_model(cfg["model_path"], len(cfg["class_names"]))
        disease_models[crop] = model
        disease_classes[crop] = cfg["class_names"]
        print(f"✅ Loaded {crop} model with {len(cfg['class_names'])} classes")
    except Exception as e:
        print(f"⚠️ Could not load {crop} model: {e}")

try:
    ALT_MODEL_PATH = "models/plant_disease_model.pth"
    ALT_CLASSES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    alternate_model = load_model(ALT_MODEL_PATH, len(ALT_CLASSES))
    print("✅ Loaded alternate fallback model")
except Exception as e:
    print(f"⚠️ Could not load alternate model: {e}")
    alternate_model = None
    ALT_CLASSES = []

try:
    yield_model = joblib.load("models/yield_pred_model.pkl")
except Exception as e:
    print(f"⚠️ Could not load yield model: {e}")
    yield_model = None

try:
    with open("disease_info.json", "r", encoding="utf-8") as f:
        DISEASE_INFO = json.load(f)
    print("✅ Loaded disease info JSON")
except Exception as e:
    print(f"⚠️ Could not load disease info JSON: {e}")
    DISEASE_INFO = {}

# Load side crops JSON
try:
    with open("data/side_crops.json", "r", encoding="utf-8") as f:
        SIDE_CROP_DATA = json.load(f)
    print("✅ Loaded side crops JSON")
except Exception as e:
    print(f"⚠️ Could not load side crops JSON: {e}")
    SIDE_CROP_DATA = []

def preprocess_image(img_path, img_size=224):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    return img.to(DEVICE)

def model_predict(image_path, crop_type):
    if crop_type in disease_models:
        model = disease_models[crop_type]
        classes = disease_classes[crop_type]
    else:
        if alternate_model is None:
            raise ValueError(f"No model available for {crop_type} and alternate model not loaded")
        print(f"⚠️ Using alternate model for unsupported crop: {crop_type}")
        model = alternate_model
        classes = ALT_CLASSES

    img = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
        prediction_label = classes[pred.item()]
    return prediction_label

def get_weather_data(state_name):
    API_KEY = "your_openweather_api_key"
    LOCATION = state_name + ",IN"

    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={LOCATION}&limit=1&appid={API_KEY}"
        geo_resp = requests.get(geo_url).json()
        if not geo_resp:
            raise ValueError("Could not geocode location")

        lat, lon = geo_resp[0]["lat"], geo_resp[0]["lon"]

        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
        forecast_resp = requests.get(forecast_url).json()

        rainfall_values = []
        for entry in forecast_resp.get("list", []):
            rain = entry.get("rain", {}).get("3h", 0.0)
            rainfall_values.append(rain)

        total_5d_rainfall = sum(rainfall_values)
        avg_daily_rainfall = total_5d_rainfall / 5.0 if rainfall_values else 0
        annual_rainfall_est = avg_daily_rainfall * 365
        avg_temp = 27

        return avg_temp, annual_rainfall_est*2
    except Exception as e:
        print("❌ Weather API Error:", e)
        return None, None

def predict_yield(crop, state, area, rainfall, fertilizer, pesticide, avg_temp):
    if yield_model is None:
        return None, None

    test_input = pd.DataFrame([{
        "Crop": crop,
        "State": state,
        "Area": area,
        "Annual_Rainfall": rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide,
        "Avg_temp": avg_temp
    }])

    yield_per_ha = yield_model.predict(test_input)[0]
    total_production = yield_per_ha * area
    return yield_per_ha, total_production

def get_soil_and_side_crop(state, district):
    """Fetch soil type and recommended side crops based on state & district"""
    for entry in SIDE_CROP_DATA:
        if entry["state"].lower() == state.lower() and entry["district"].lower() == district.lower():
            soil_type = entry.get("soil_type", "Unknown")
            side_crops = entry.get("side_crops", [])
            return soil_type, side_crops
    return "Unknown", []

# ------------- ROUTES ------------- #

@app.route('/visualize_yield')
@login_required
def visualize_yield():
    analyses = CropData.query.filter_by(user_id=current_user.id).order_by(CropData.created_at).all()
    if not analyses:
        flash("No data to visualize")
        return redirect(url_for('dashboard'))

    dates = [a.created_at.strftime('%Y-%m-%d') for a in analyses]
    yields = []
    for a in analyses:
        if a.yield_prediction:
            try:
                yield_per_ha = float(a.yield_prediction.split()[0])
                yields.append(yield_per_ha)
            except:
                yields.append(0)
        else:
            yields.append(0)

    plt.figure(figsize=(8, 4))
    plt.plot(dates, yields, marker='o', color='blue')
    plt.title("Yield Predictions Over Time")
    plt.xlabel("Date")
    plt.ylabel("Yield (t/ha)")
    plt.xticks(rotation=45)
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    yield_chart = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    plt.close()

    labels = {}
    for a in analyses:
        disease = a.disease_name or "Unknown"
        labels[disease] = labels.get(disease, 0) + 1

    plt.figure(figsize=(6, 6))
    plt.pie(labels.values(), labels=labels.keys(), autopct='%1.1f%%', startangle=140)
    plt.title("Disease Distribution")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    disease_chart = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    plt.close()

    crop_yields = {}
    for a in analyses:
        if a.yield_prediction:
            try:
                yield_per_ha = float(a.yield_prediction.split()[0])
                crop_yields.setdefault(a.crop_type, []).append(yield_per_ha)
            except:
                continue

    crops = list(crop_yields.keys())
    avg_yields = [np.mean(vals) for vals in crop_yields.values()]

    plt.figure(figsize=(7, 5))
    plt.bar(crops, avg_yields, color='green')
    plt.title("Average Yield by Crop Type")
    plt.xlabel("Crop")
    plt.ylabel("Yield (t/ha)")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    crop_chart = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    plt.close()

    return render_template('visual_yield.html',
                           yield_chart=yield_chart,
                           disease_chart=disease_chart,
                           crop_chart=crop_chart)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username already exists!')
            return redirect(url_for('signup'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered!')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Signup successful! Please login.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form['username']
        password = request.form['password']

        user = User.query.filter(
            (User.username == identifier) | (User.email == identifier)
        ).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    analyses = (CropData.query
                .filter_by(user_id=current_user.id)
                .order_by(CropData.created_at.desc())
                .all())
    total_analyses = len(analyses)
    healthy_plants = sum(1 for a in analyses if a.disease_status == 0)
    diseases_detected = total_analyses - healthy_plants
    recent_analyses = analyses[:5]
    return render_template('dashboard.html',
                           username=current_user.username,
                           total_analyses=total_analyses,
                           healthy_plants=healthy_plants,
                           diseases_detected=diseases_detected,
                           recent_analyses=recent_analyses)

@app.route("/analyze")
@login_required
def analyze():
    return render_template("analyze.html")

@app.route('/upload/', methods=['POST', 'GET'])
@login_required
def uploadimage():
    if request.method == "POST":
        if 'img' not in request.files:
            flash("No file part in request")
            return redirect(request.url)

        image = request.files['img']
        if image.filename == '':
            flash("No selected file")
            return redirect(request.url)

        filename = secure_filename(image.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        image.save(save_path)

        crop_type = request.form.get("crop_type")
        state = request.form.get("state")
        district = request.form.get("district")
        area = float(request.form.get("area") or 0)
        fertilizer = float(request.form.get("fertilizer") or 0)
        pesticide = float(request.form.get("pesticide") or 0)

        if not crop_type or not state or not district:
            flash("Please provide crop type, state and district")
            return redirect(request.url)

        try:
            prediction = model_predict(save_path, crop_type)
        except Exception as e:
            print("Prediction error:", e)
            prediction = "Not available"

        if "healthy" in str(prediction).lower():
            disease_status = 0
        else:
            disease_status = 1

        avg_temp, rainfall = get_weather_data(state)
        if avg_temp is None:
            avg_temp, rainfall = None, None
            
        try:
            yield_per_ha, total_production = predict_yield(
                crop=crop_type,
                state=state,
                area=area,
                rainfall=rainfall or 0,
                fertilizer=fertilizer,
                pesticide=pesticide,
                avg_temp=avg_temp or 0
            )
        except:
            yield_per_ha, total_production = None, None

        # Fetch soil type and side crops
        soil_type, recommended_side_crops = get_soil_and_side_crop(state, district)

        new_analysis = CropData(
            crop_type=crop_type,
            disease_status=disease_status,
            disease_name=prediction,
            yield_prediction=f"{yield_per_ha:.2f} t/ha ({total_production:.2f} tonnes)" if yield_per_ha is not None else None,
            user_id=current_user.id
        )
        db.session.add(new_analysis)
        db.session.commit()

        disease_cause = None
        disease_cure = None
        try:
            if crop_type in DISEASE_INFO and prediction in DISEASE_INFO[crop_type]:
                disease_cause = DISEASE_INFO[crop_type][prediction]["cause"]
                disease_cure = DISEASE_INFO[crop_type][prediction]["cure"]
            elif prediction in DISEASE_INFO.get("multiple_plants", {}):
                disease_cause = DISEASE_INFO["multiple_plants"][prediction]["cause"]
                disease_cure = DISEASE_INFO["multiple_plants"][prediction]["cure"]
        except Exception as e:
            print("⚠️ Error fetching disease info:", e)

        return render_template("final.html",
            image_url=url_for('static', filename=f"uploads/{unique_filename}"),
            prediction=prediction,
            yield_prediction=f"{yield_per_ha:.2f} t/ha ({total_production:.2f} tonnes)" if yield_per_ha is not None else None,
            yield_per_ha=yield_per_ha,
            total_production=total_production,
            avg_temp=avg_temp,
            rainfall=rainfall,
            username=current_user.username,
            disease_cause=disease_cause,
            disease_cure=disease_cure,
            soil_type=soil_type,
            recommended_side_crops=recommended_side_crops,
            state=state,         
            district=district 
        )

    return redirect('/')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

## 🌾 Smart Crop Monitoring & Yield Prediction System
An AI-powered system for real-time crop health monitoring and accurate yield prediction

## 📌 Overview
This project is a full-stack agricultural intelligence system built using:

Flask (Backend + Auth + Dashboard)
PyTorch (Disease Classification Models)
Scikit-Learn (Yield Prediction Model)
OpenWeatherMap API (Weather-based yield estimation)
SQLite (User Database)
Bootstrap/UI Templates (Frontend)

Users can:
✔ Upload a crop image
✔ Detect crop diseases using CNN models
✔ Predict yield based on environmental inputs
✔ View charts & insights
✔ Manage login/signup
✔ Store history of analyses

## 🚀 Key Features
🌱 1. Crop Disease Detection

Supports multiple crops:
Rice, Wheat, Maize, Onion, Soyabean, Millet, Cotton, Sugarcane
Each crop has its own MobileNetV2-based classifier
Fallback “generic model” for other crops
Uses PyTorch models (*.pth)

📊 2. Smart Yield Prediction

Uses ML model (yield_pred_model.pkl) based on:
Crop
State
Area
Fertilizer
Pesticide
Rainfall (pulled automatically via weather API)
Avg temperature
Outputs:
Yield per hectare
Total expected production

☁️ 3. Weather Integration (OpenWeatherMap API)

Auto-fetches rainfall + temperature
Uses 5-day rainfall forecast
Converts to yearly rainfall estimation

🔐 4. User Auth System

Login / Signup
Secure password hashing
Flask-Login based session management
Individual user dashboards

📈 5. Data Visualization

Includes:
Yield over time
Disease distribution pie chart
Average crop yield bar chart
Rendered with matplotlib and embedded into HTML using Base64.

## 📂 Project Structure
project/
│
├── app.py                           # Main Flask backend
├── models/
│   ├── *_disease_model.pth          # Torch models
│   └── yield_pred_model.pkl         # Yield prediction ML model
│
├── static/
│   ├── uploads/                     # Uploaded images
│   └── css/js                       # UI assets
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── dashboard.html
│   ├── analyze.html
│   ├── final.html
│   └── visual_yield.html
│
├── disease_info.json                # Disease cause/cure database
└── README.md

## 📥 Download Yield Prediction Model

The trained **yield prediction model (yield_pred_model.pkl)** is stored on Google Drive.

👉 **Download here:**  
[📦 Click to Download Yield Prediction Model](https://drive.google.com/uc?export=download&id=1UCFj1Q7BGfLdMHBquT9IxoN-dRtcvq9e)

> Place the downloaded file in the `/models` directory.

## 🌾 Smart Crop Monitoring & Yield Prediction System
An AI-powered platform for early crop disease detection and accurate agricultural yield prediction using machine learning and real-time weather data.

---

## 📌 Overview
This project is a full-stack agricultural intelligence system designed to assist farmers, researchers, and analysts in monitoring crop health and estimating yield through data-driven insights.

The system integrates computer vision, machine learning, weather data, and secure user management to provide a complete end-to-end solution for smart agriculture.

---

## 🛠 Tech Stack
- **Backend:** Flask (Python)
- **Machine Learning:** PyTorch, scikit-learn
- **Database:** SQLite
- **Frontend:** HTML, CSS, Bootstrap
- **APIs:** OpenWeatherMap API
- **Visualization:** Matplotlib

---

## 👤 User Capabilities
Users can:
- Upload crop images for analysis
- Detect crop diseases using CNN models
- Predict crop yield based on environmental and agricultural inputs
- View analytical charts and insights
- Register and log in securely
- Store and view past analysis history

---

## 🚀 Key Features

### 1. Crop Disease Detection
- Supports multiple crops:
  **Rice, Wheat, Maize, Onion, Soyabean, Millet, Cotton, Sugarcane**
- Each crop has a dedicated **MobileNetV2-based CNN model**
- Includes a fallback **generic disease detection model**
- Models trained using transfer learning on labeled crop disease datasets
- Implemented using **PyTorch (.pth files)**

---

### 2. Smart Yield Prediction
- Machine learning model: `yield_pred_model.pkl`
- Inputs include:
  - Crop type
  - State
  - Cultivated area
  - Fertilizer usage
  - Pesticide usage
  - Rainfall (auto-fetched via weather API)
  - Average temperature
- Outputs:
  - Yield per hectare
  - Total expected production

---

### 3. Weather Integration (OpenWeatherMap API)
- Automatically fetches:
  - Rainfall data
  - Temperature data
- Uses 5-day forecast data
- Converts forecast into estimated annual rainfall for prediction

---

### 4. User Authentication System
- Login and Signup functionality
- Secure password hashing
- Flask-Login based session management
- Individual user dashboards with stored analysis history

---

### 5. Data Visualization & Insights
- Yield trends over time
- Disease distribution (pie chart)
- Average crop yield comparison (bar chart)
- Visualizations rendered using **Matplotlib**
- Charts embedded into HTML using **Base64 encoding**

---

## 📂 Project Structure
project/
│
├── app.py # Main Flask backend
├── models/
│ ├── *_disease_model.pth # PyTorch disease detection models
│ └── yield_pred_model.pkl # Yield prediction ML model
│
├── static/
│ ├── uploads/ # Uploaded crop images
│ └── css/js # Frontend assets
│
├── templates/
│ ├── index.html
│ ├── login.html
│ ├── signup.html
│ ├── dashboard.html
│ ├── analyze.html
│ ├── final.html
│ └── visual_yield.html
│
├── disease_info.json # Disease causes & cure information
└── README.md


---

## 📥 Download Yield Prediction Model
The trained **yield prediction model (`yield_pred_model.pkl`)** is hosted externally due to size constraints.

👉 **Download here:**  
[📦 Click to Download Yield Prediction Model](https://drive.google.com/uc?export=download&id=1UCFj1Q7BGfLdMHBquT9IxoN-dRtcvq9e)

> After downloading, place the file inside the `/models` directory.

---

## 👥 Contributors
- **Girish Nalkar** –  Machine learning model training, data preprocessing, and frontend components
- **Abhiram Nair** – Backend architecture, API development, authentication, ML model integration, and frontend workflows  
- **Divyanshi** – Backend development, database handling, API testing, and frontend feature implementation  
- **Jasleen** – ML model development, experimentation, optimization, and frontend support

---

## 📈 Learning Outcomes
This project strengthened understanding of:
- Full-stack application architecture
- RESTful API development
- Machine learning model integration
- Secure authentication systems
- Real-world data processing and visualization
- End-to-end ML-powered application design

# 🌾 Crop Yield Prediction Web App

A **Flask web app** to predict agricultural crop yield based on climate, soil, and crop parameters using machine learning models.

---

## 🚀 Features

🔢 **Input Parameters**  
Enter details like region, soil type, rainfall, temperature, and crop to get yield predictions.

🤖 **Multiple ML Models Used**  
- Linear Regression  
- Ridge Regression  
- Gradient Boosting  
- LightGBM  
- CatBoost
- XGBoost

📈 **Prediction Comparison**  
Get predictions from all models and compare their outputs instantly.

🧪 **Real-time**  
- Submit form for real-time yield prediction  

📊 **Clean UI with Form Controls**  
Dropdowns for categorical fields, toggle switches for boolean options, and input boxes for numeric values.

---

## 📸 Demo

### 📝 Input Form
![form](<img width="1138" height="599" alt="image" src="https://github.com/user-attachments/assets/503cb3d2-309e-48d5-a474-da48254c0587" />
)

### 📊 Prediction Output
![output](<img width="1193" height="370" alt="image" src="https://github.com/user-attachments/assets/6040d00f-a016-4370-bf73-6e734b4fba55" />
)


---

## 🧰 Technologies Used

- 🐍 Python  
- 🔥 Flask  
- 📦 Scikit-learn, Pandas, NumPy  
- 📊 LightGBM, CatBoost, Gradient Boosting  
- 🎨 HTML, CSS  
- 🎯 Bootstrap for UI styling

---

## 📁 Input Parameters

| Parameter | Type | Options / Example |
|----------|------|-------------------|
| Region | Dropdown | North, South, East, West |
| Soil_Type | Dropdown | Clay, Sandy, Loam, Silt, Peaty, Chalky |
| Crop | Dropdown | Wheat, Rice, Maize, Barley, Soybean, Cotton |
| Rainfall_mm | Numeric | e.g., 350 |
| Temperature_Celsius | Numeric | e.g., 27 |
| Fertilizer_Used | Checkbox | Yes / No |
| Irrigation_Used | Checkbox | Yes / No |
| Weather_Condition | Dropdown | Sunny, Rainy, Cloudy |
| Days_to_Harvest | Numeric | e.g., 120 |

---

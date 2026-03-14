# Medical Insurance Cost Prediction

## Project Overview

This project builds a machine learning model to predict individual medical insurance charges based on demographic and health-related attributes such as age, BMI, smoking status, number of children, gender, and region.

The objective is to understand the factors that influence healthcare costs and build a predictive model capable of estimating insurance charges accurately.

---

## Dataset

The dataset contains the following features:

- **age** – age of the individual
- **sex** – gender
- **bmi** – body mass index
- **children** – number of dependents
- **smoker** – smoking status
- **region** – residential region
- **charges** – medical insurance cost (target variable)

---

## Exploratory Data Analysis (EDA)

The exploratory analysis revealed several key insights:

- Medical charges are **right-skewed**, meaning most individuals have moderate costs while a few have extremely high expenses.
- **Age** shows a strong positive relationship with insurance charges.
- **Smoking status** significantly increases medical costs.
- Higher **BMI**, especially when combined with smoking, leads to much higher charges.
- **Region and sex** have relatively small influence on insurance costs.

---

## Feature Engineering

Based on insights from the EDA, several engineered features were created:

- **age_squared** – captures nonlinear age effects
- **bmi_obese** – indicator for BMI ≥ 30
- **age_smoker** – interaction between age and smoking
- **bmi_smoker** – interaction between BMI and smoking

These features help models capture nonlinear relationships and interaction effects.

---

## Models Evaluated

The following models were trained and evaluated:

| Model | R² | MAE |
|------|------|------|
| Linear Regression | 0.807 | 4177 |
| Engineered Linear Regression | 0.824 | 2956 |
| Random Forest | 0.881 | 2647 |
| Engineered Random Forest | 0.893 | 2083 |
| XGBoost | 0.852 | 2704 |
| **Tuned Random Forest** | **0.898** | **1937** |

The **tuned Random Forest model** achieved the best performance.

---

## Final Model Performance

Final tuned Random Forest metrics:

- **R²:** 0.898  
- **MAE:** \$1,937  
- **RMSE:** ~\$4,324  

This means the model explains nearly **90% of the variance in insurance charges**.

---

## Feature Importance

The most influential predictors were:

1. Age
2. Age² (nonlinear age effect)
3. BMI × Smoker interaction
4. Age × Smoker interaction
5. Smoking status

Features such as **sex and region had minimal impact** on predictions.

---

## Project Structure
Medical insurance/
│
├── 01_EDA.ipynb
├── 02_Modeling.ipynb
├── insurance.csv
├── cleaned_insurance_data.csv
├── tuned_random_forest_model.pkl
├── model_features.pkl
├── requirements.txt
└── README.md

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## Conclusion

This project demonstrates how exploratory data analysis, feature engineering, and ensemble learning methods can effectively model healthcare costs.

The tuned Random Forest model achieved strong predictive performance and revealed that **age, smoking behavior, and BMI interactions are the primary drivers of insurance charges**.

## Using the Trained Model

The final tuned Random Forest model has been saved and can be loaded for predictions.

### Load the Model

```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load("tuned_random_forest_model.pkl")
features = joblib.load("model_features.pkl")

Example Prediction

sample = pd.DataFrame({
    "age": [45],
    "sex": [1],
    "bmi": [32.5],
    "children": [2],
    "smoker": [1],
    "region_northwest": [0],
    "region_southeast": [1],
    "region_southwest": [0],
    "age_smoker": [45],
    "bmi_smoker": [32.5],
    "age_squared": [2025],
    "bmi_obese": [1]
})

prediction_log = model.predict(sample)
prediction = np.expm1(prediction_log)

print("Predicted Insurance Charge:", prediction[0])

Future Improvements

Possible future improvements for this project include:

Building a prediction API using FastAPI

Creating an interactive Streamlit dashboard

Performing advanced hyperparameter optimization

Exploring additional ensemble models
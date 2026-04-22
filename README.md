##  Files Included

- Dinesh_Dissertation_F1.ipynb → Main implementation notebook  
- README.md → Project description  


#  Formula 1 Race Outcome Prediction using Machine Learning

##  Overview
This project aims to predict Formula 1 race outcomes using machine learning techniques.  
The study focuses on two main tasks:
- Classification: Predict whether a driver finishes in the top 3
- Regression: Predict the final race position  

The models are trained using pre-race variables such as grid position and qualifying times.

---

##  Dataset
The dataset consists of historical Formula 1 race data, including:

- Grid Position  
- Qualifying Times (Q1, Q2, Q3)  
- Final Race Position  

The dataset was cleaned and preprocessed before modelling.

---

## Machine Learning Models Used

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  

---

##  Results

### Classification:
- Accuracy: ~81.5%  
- AUC Score: 0.81  

### Regression:
- Low R² score (~0.14) indicating limited predictive power  

---

## Key Findings

- Grid position is the most important feature  
- Classification models perform better than regression  
- Complex models do not always outperform simpler models  
- Lack of race dynamics limits prediction accuracy  

##  Visual Results

The notebook includes:
- Confusion Matrix  
- ROC Curve  
- Feature Importance Plot  
---


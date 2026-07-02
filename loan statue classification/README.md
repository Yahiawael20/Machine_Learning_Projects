#  Credit Risk Prediction Project

This project focuses on building a **Credit Risk Prediction system** using Machine Learning techniques, along with applying **MLOps practices** for model development, evaluation, and deployment.

---

##  Project Overview
The goal of this project is to predict whether a customer is **Low Risk** or **High Risk** based on financial and demographic features.

- Data preprocessing and cleaning
- Feature engineering
- Model training and evaluation
- Model selection and saving
- Deployment using Streamlit

---

##  Technologies Used
- Python 
- Pandas & NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Streamlit
- Joblib / Pickle

---

##  Models Implemented
- Logistic Regression
- Random Forest
- XGBoost
- Other baseline models

🔹 Applied **Grid Search** for hyperparameter tuning  
🔹 Compared models and selected the best-performing one  

---

##  Model Performance
- **Accuracy:** 93%  
- **Recall (High Risk):** 74%  
- **F1-score:** 83%  

---

##  Challenges
### Data Imbalance
The dataset suffers from imbalance, especially in the **High Risk class**, which affects the model’s ability to correctly identify risky customers.

- Tried **SMOTE** to handle imbalance  
- Limited improvement observed  
- Highlighted the importance of using Recall & F1-score instead of Accuracy alone  

---

##  MLOps Workflow
- Built a structured ML pipeline  
- Model training & evaluation pipeline  
- Model selection and saving  
- Ready for deployment (Streamlit app)  

---

##  Key Takeaways
- Accuracy is not enough for imbalanced datasets  
- Recall is critical in risk-sensitive applications  
- SMOTE and similar techniques may not always solve imbalance issues  
- Proper evaluation metrics are essential  
- Collecting or sourcing **new data for the minority class** can significantly improve model performance  


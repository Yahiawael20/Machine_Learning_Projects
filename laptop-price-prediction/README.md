#  Laptop Price Prediction

This project is a Machine Learning model that predicts laptop prices based on different specifications using Python and Scikit-learn. It is deployed using Streamlit for an interactive user experience.

---

##  Project Overview
The goal of this project is to estimate laptop prices based on features such as:
- Company
- Type (Gaming / Notebook / Ultrabook, etc.)
- RAM
- Weight
- Touchscreen
- IPS Display
- CPU & GPU
- Operating System

---

##  Machine Learning Models and Techniques Used
- Linear Regression
- Random Forest Regressor
- XGBoost
- Grids Search
  

---

##  Data Preprocessing
- Handling categorical variables using One-Hot Encoding
- Feature scaling 
- Removing missing / inconsistent values


---

##  Tech Stack
- Python 
- Pandas & NumPy
- Scikit-learn
- Streamlit (for deployment)
- Matplotlib / Seaborn (For Visualization)

---

##  How to Run the Project

```bash id="7kqv1a"
git clone https://github.com/your-username/laptop-price-prediction.git
cd laptop-price-prediction
pip install -r requirements.txt
streamlit run app.py

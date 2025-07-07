import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Title
st.title(" Titanic Survival Predictor")
st.write("Enter passenger details to predict survival")

# Sidebar inputs
Pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 1, 80, 25)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
Parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
Fare = st.slider("Fare Paid", 0.0, 500.0, 50.0)
Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Encode inputs
sex_map = {"male": 1, "female": 0}
embarked_map = {"S": 2, "C": 0, "Q": 1}

# Load and prepare training data
@st.cache_data
def train_models():
    data = pd.read_csv("train.csv")
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

    le = LabelEncoder()
    data['Sex'] = le.fit_transform(data['Sex'])
    data['Embarked'] = le.fit_transform(data['Embarked'])

    X = data.drop('Survived', axis=1)
    y = data['Survived']

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X, y)

    return rf_model, xgb_model

rf_model, xgb_model = train_models()

# Collect inputs into a dataframe
input_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Sex': [sex_map[Sex]],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Embarked': [embarked_map[Embarked]]
})

# Predict
if st.button("Predict with Random Forest"):
    pred = rf_model.predict(input_data)[0]
    st.success(" Survived" if pred == 1 else " Did not survive")

if st.button("Predict with XGBoost"):
    pred = xgb_model.predict(input_data)[0]
    st.success(" Survived" if pred == 1 else " Did not survive")

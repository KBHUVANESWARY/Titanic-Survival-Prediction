import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission_template = pd.read_csv("gender_submission.csv")  # for format

# Combine train and test for feature engineering
test['Survived'] = np.nan  # Add dummy target column
combined = pd.concat([train, test], ignore_index=True)

# ----------------------------
# 🧠 Feature Engineering
# ----------------------------

# Title from Name
combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
combined['Title'] = combined['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined['Title'] = combined['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# Fill missing values
combined['Embarked'].fillna(combined['Embarked'].mode()[0], inplace=True)
combined['Fare'].fillna(combined['Fare'].median(), inplace=True)
combined['Age'].fillna(combined['Age'].median(), inplace=True)

# Binning Fare and Age (convert to int to avoid category dtype error)
combined['FareBand'] = pd.qcut(combined['Fare'], 4, labels=[0, 1, 2, 3]).astype(int)
combined['AgeBand'] = pd.cut(combined['Age'], 5, labels=[0, 1, 2, 3, 4]).astype(int)

# Family Size
combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1

# Encode categorical features
label_cols = ['Sex', 'Embarked', 'Title']
for col in label_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

# Drop unused columns
combined.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare', 'Age'], axis=1, inplace=True)

# ----------------------------
# 🔀 Split train/test data
# ----------------------------
train_final = combined[combined['Survived'].notnull()]
test_final = combined[combined['Survived'].isnull()].drop('Survived', axis=1)

X = train_final.drop('Survived', axis=1)
y = train_final['Survived'].astype(int)

# ----------------------------
# ⚙️ XGBoost + GridSearchCV
# ----------------------------
params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

grid = GridSearchCV(estimator=xgb, param_grid=params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid.fit(X, y)

print("✅ Best Parameters:", grid.best_params_)
print("✅ Best CV Accuracy:", round(grid.best_score_ * 100, 2), "%")

# ----------------------------
# 📤 Predict and Save Submission
# ----------------------------
final_model = grid.best_estimator_
predictions = final_model.predict(test_final)

# Create final submission.csv
submission = pd.DataFrame({
    'PassengerId': pd.read_csv("test.csv")['PassengerId'],
    'Survived': predictions.astype(int)
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv generated successfully!")

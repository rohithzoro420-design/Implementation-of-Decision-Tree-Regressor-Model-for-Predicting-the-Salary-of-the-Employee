# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1) Load & Prepare Data – Read the Salary dataset, encode the categorical Position column using LabelEncoder, and separate it into features X (Position, Level) and target y (Salary).
2) Split the Data – Divide X and y into training and testing sets using train_test_split().
3) Train the Model – Create a DecisionTreeRegressor object and fit it on the training data (X_train, Y_train).
4) Predict & Evaluate – Use the trained model to predict salaries on the test set, evaluate performance using the R² score, and visualize the tree using plot_tree().
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Rohith S
RegisterNumber:  212225240122
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

csv_path = r"D:\Salary.csv"    
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at: {csv_path}. Update the path.")

print("Dataset Loaded Successfully!\n")

print("Shape:", data.shape)
print(data.head())

print("\nInfo:")
print(data.info())

print("\nMissing Values:\n", data.isnull().sum())

if "Position" in data.columns:
    le = LabelEncoder()
    data["Position"] = le.fit_transform(data["Position"])
    print("\nLabel Encoding Mapping (Position):")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)

X = data[["Position", "Level"]]
y = data["Salary"]

print("\nFeature Sample:")
print(X.head())

print("\nTarget Sample:")
print(y.head())

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
print(f"\nTrain Size: {X_train.shape}, Test Size: {X_test.shape}")

dt = DecisionTreeRegressor(random_state=10)
dt.fit(X_train, Y_train)
print("\nModel Training Completed!")

y_pred = dt.predict(X_test)
print("\nPredicted Salaries:", y_pred)

r2 = metrics.r2_score(Y_test, y_pred)
print(f"\nR2 Score: {r2:.4f}")

plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=["Position", "Level"], filled=True)
plt.title("Decision Tree Regressor for Salary Prediction")
plt.tight_layout()
plt.savefig("decision_tree_plot.png", dpi=150)
print("\nDecision tree plot saved as decision_tree_plot.png")

importances = pd.Series(dt.feature_importances_, index=["Position", "Level"])
print("\nFeature Importances:")
print(importances)

```

## Output:

<img width="1237" height="602" alt="image" src="https://github.com/user-attachments/assets/a7df2e1f-807f-417d-9b1d-0abd514de370" />

<img width="1568" height="731" alt="image" src="https://github.com/user-attachments/assets/f48013dd-b1ca-415f-9251-e052ee8e482e" />

<img width="1448" height="726" alt="image" src="https://github.com/user-attachments/assets/686db902-ce6e-497c-8099-5229accbb23c" />

<img width="1465" height="687" alt="image" src="https://github.com/user-attachments/assets/6cfffcfc-f1aa-4b99-ad11-fc1d6c4b715c" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

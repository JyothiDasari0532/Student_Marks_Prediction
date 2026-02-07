# Student_Marks_Prediction
A simple Machine Learning project that predicts student marks based on study hours using Linear Regression in Python.
code :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [10,20,30,40,50,60,70,80,85,95]
}

df = pd.DataFrame(data)

X = df[['Hours']]
Y = df['Marks']

model = LinearRegression()
model.fit(X, Y)

hours = float(input("Enter study hours: "))

predicted_marks = model.predict([[hours]])

print("Predicted Marks:", round(predicted_marks[0], 2))

plt.scatter(X, Y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.show()

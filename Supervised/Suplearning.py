# This is a test program

# Import statements 
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score # if R2 close 1, then the model is good

# Read in the data.
data = pd.read_csv("data1.csv")
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Use a test size of 25% and a random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#LinearRegression
LRclassifier = LinearRegression()
LRclassifier.fit(X,y)
guesses = LRclassifier.predict(X)
error = mean_squared_error(y, guesses)
print("LinearRegression: mean_squared_error = ", error)
print("LinearRegression: r2 score = ", r2_score(y, guesses))

# Decision tree
DTclassifier = DecisionTreeClassifier()
DTclassifier.fit(X,y)
guesses = DTclassifier.predict(X)
error = mean_squared_error(y, guesses)
print("DecisionTree: mean_squared_error = ", error)
print("DecisionTree: r2 score = ", r2_score(y, guesses))

# with split data (train and test)
# Fit the model to the training data.
DTclassifier.fit(X_train,y_train)
# Make predictions on the test data
y_pred = DTclassifier.predict(X_test)

# Print actual Vx predicted
for i , j in zip (y_test, y_pred):
	print("actual y = ", i , "predicted y = ", j)

# Logistic tree
Logisticclassifier = LogisticRegression()
Logisticclassifier.fit(X,y)
guesses = Logisticclassifier.predict(X)
error = mean_squared_error(y, guesses)
print("Logistic: mean_squared_error = ", error)
print("Logistic: r2 score = ", r2_score(y, guesses))


# SVM
SVMclassifier = SVC()
SVMclassifier.fit(X,y)
guesses = SVMclassifier.predict(X)
error = mean_squared_error(y, guesses)
print("SVM: mean_squared_error = ", error)
print("SVM: r2 score = ", r2_score(y, guesses))


# Neural network
#from sklearn.neural_network import MLPClassifier
#NNclassifier = MLPClassifier()
#NNclassifier.fit(X,y)



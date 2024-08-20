import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_data = pd.read_csv('diabetes.csv')

print(diabetes_data.head())
print()
print(diabetes_data.describe())
print()
print(diabetes_data['Outcome'].value_counts())
print()
print(diabetes_data.groupby('Outcome').mean())

#seperating data and lebels
X = diabetes_data.drop(columns = 'Outcome', axis=1)
Y = diabetes_data['Outcome']

#data Standaerdization    data similar range e nilam
scalar = StandardScaler()
scalar.fit(X)
standardized_data = scalar.transform(X)
#print(standardized_data)

X = standardized_data

#TrainTest
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#train model

classifier  = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

#Model Eval
#testing train data kotota valo kaaj kore
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_prediction) 
print('Accuracy on training data : ', training_data_accuracy)

#testing test data 
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test,X_test_prediction) 
print('Accuracy on test data : ', test_data_accuracy)


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scalar.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
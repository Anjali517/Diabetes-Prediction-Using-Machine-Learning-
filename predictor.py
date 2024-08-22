#Importing the dependencies


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#Accuracy= Total Number of Predictions/ Number of Correct Predictions​

#accuracy_score is a function from the sklearn.metrics module in scikit-learn. 
#It is used to compute the accuracy of a classification model, which is the ratio of correctly predicted samples to the total number of samples.
  
#data collection and analysis
diabetes_dataset=pd.read_csv('/content/diabetes.csv')

#print first 5 rows
diabetes_dataset.head()
bmi=weight/height^2
#The DiabetesPedigreeFunction is a feature that provides a summary of diabetes history in relatives and the genetic relationship of those relatives to the patient. 
#It is a numeric value that was computed based on the family history of diabetes and the degree of relation to the individual being studied.
#Essentially, it indicates the likelihood of a person developing diabetes based on their family history.

diabetes_dataset.shape
diabetes_dataset.describe()
diabetes_dataset['Insulin'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
#separating the data and labels
#axis=1 for dropping column and 0 for a row
x = diabetes_dataset.drop(columns = 'Outcome',axis=1)
y=diabetes_dataset['Outcome']
print(x)
print(y)
scaler=StandardScaler()
scaler.fit(x)
standardized_data=scaler.transform(x)
print(standardized_data)
x = standardized_data
y = diabetes_dataset['Outcome']
print(x)
print(y)
x_train , x_test ,y_train ,y_test = train_test_split(x,y,test_size=0.2, stratify=y, random_state=2)
print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
#training the model
classifier = svm.SVC(kernel='linear')
training the svm classifier
classifier.fit(x_train,y_train)

#Model Evaluation

x_train_pred=classifier.predict(x_train)
train_acc=accuracy_score(x_train_pred,y_train)
print(train_acc)

x_test_pred=classifier.predict(x_test)
test_acc=accuracy_score(x_test_pred,y_test)

print(test_acc)

#Making a Predictive System



input_data=(1,85,66,29,0,26.6,0.351,31)
#change input data to numpy array cause processing is easy in numpy
input_array_as_numpy_array=np.asarray(input_data)

#reshape the array for prediction (for specifying we need prediction for only one row)
input_data_reshaped=input_array_as_numpy_array.reshape(1,-1)

#standradize input data
std_data=scaler.transform(input_data_reshaped)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)

#prediction is a list

if prediction==0:
  print("The person is not diabetic")
else:
  print("The person is diabetic")


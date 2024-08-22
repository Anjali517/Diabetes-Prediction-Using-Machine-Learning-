#Importing the dependencies


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
Accuracy= Total Number of Predictions/ Number of Correct Predictions​

#accuracy_score is a function from the sklearn.metrics module in scikit-learn. 
#It is used to compute the accuracy of a classification model, which is the ratio of correctly predicted samples to the total number of samples.

data collection and analysis



diabetes_dataset=pd.read_csv('/content/diabetes.csv')


#print first 5 rows
diabetes_dataset.head()


bmi=weight/height^2

#The DiabetesPedigreeFunction is a feature that provides a summary of diabetes history in relatives and the genetic relationship of those relatives to the patient. 
#It is a numeric value that was computed based on the family history of diabetes and the degree of relation to the individual being studied.
#Essentially, it indicates the likelihood of a person developing diabetes based on their family history.


diabetes_dataset.shape
(768, 9)


diabetes_dataset.describe()



diabetes_dataset['Insulin'].value_counts()



diabetes_dataset.groupby('Outcome').mean()



#separating the data and labels
#axis=1 for dropping column and 0 for a row
x = diabetes_dataset.drop(columns = 'Outcome',axis=1)
y=diabetes_dataset['Outcome']


print(x)
     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
0              6      148             72             35        0  33.6   
1              1       85             66             29        0  26.6   
2              8      183             64              0        0  23.3   
3              1       89             66             23       94  28.1   
4              0      137             40             35      168  43.1   
..           ...      ...            ...            ...      ...   ...   
763           10      101             76             48      180  32.9   
764            2      122             70             27        0  36.8   
765            5      121             72             23      112  26.2   
766            1      126             60              0        0  30.1   
767            1       93             70             31        0  30.4   

     DiabetesPedigreeFunction  Age  
0                       0.627   50  
1                       0.351   31  
2                       0.672   32  
3                       0.167   21  
4                       2.288   33  
..                        ...  ...  
763                     0.171   63  
764                     0.340   27  
765                     0.245   30  
766                     0.349   47  
767                     0.315   23  

[768 rows x 8 columns]


print(y)
0      1
1      0
2      1
3      0
4      1
      ..
763    0
764    0
765    0
766    1
767    0
Name: Outcome, Length: 768, dtype: int64
standardise the data



scaler=StandardScaler()


scaler.fit(x)



standardized_data=scaler.transform(x)


print(standardized_data)
[[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
   1.4259954 ]
 [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
  -0.19067191]
 [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
  -0.10558415]
 ...
 [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
  -0.27575966]
 [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
   1.17073215]
 [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
  -0.87137393]]


x = standardized_data
y = diabetes_dataset['Outcome']


print(x)
print(y)
[[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
   1.4259954 ]
 [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
  -0.19067191]
 [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
  -0.10558415]
 ...
 [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
  -0.27575966]
 [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
   1.17073215]
 [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
  -0.87137393]]
0      1
1      0
2      1
3      0
4      1
      ..
763    0
764    0
765    0
766    1
767    0
Name: Outcome, Length: 768, dtype: int64


x_train , x_test ,y_train ,y_test = train_test_split(x,y,test_size=0.2, stratify=y, random_state=2)


print(x.shape)
(768, 8)


print(x_train.shape)
(614, 8)


print(x_test.shape)
(154, 8)


print(y_train.shape)
(614,)


#training the model


classifier = svm.SVC(kernel='linear')
training the svm classifier



classifier.fit(x_train,y_train)

Model Evaluation



x_train_pred=classifier.predict(x_train)
train_acc=accuracy_score(x_train_pred,y_train)


print(train_acc)
0.7866449511400652


x_test_pred=classifier.predict(x_test)
test_acc=accuracy_score(x_test_pred,y_test)


print(test_acc)
0.7727272727272727
Making a Predictive System



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

if prediction[0]==0:
  print("The person is not diabetic")
else:
  print("The person is diabetic")


[[-0.84488505 -1.12339636 -0.16054575  0.53090156 -0.69289057 -0.68442195
  -0.36506078 -0.19067191]]
[0]

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn

df = pd.read_csv('heart.csv')

df['sex'] = df['sex'].replace(1,'Male')
df['sex'] = df['sex'].replace(0,'Female')

df['exang'] = df['exang'].replace(1,'Yes')
df['exang'] = df['exang'].replace(0,'No')

df.columns
X=df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']]

y = df['target']

X.isnull().sum()

X['trestbps'].hist()

for i in X['trestbps']:
    if(i>=120 and i<=140):
        X['trestbps']=X['trestbps'].replace(i,'Normal')
    elif(i<120):
        X['trestbps']=X['trestbps'].replace(i,'LowBP')
    else:
        X['trestbps']=X['trestbps'].replace(i,'HighBP')

X['age'].hist()
X['age'].skew(axis=0, skipna = 'True')

X['age'].max()    
X['age'].min()    
X['age'].mode()    
X['age'].mean()    

X['age'].groupby(X['sex']).plot()

X['chol'].hist()
X['chol'].skew(axis=0, skipna = 'True')
X['thalach'].skew(axis=0, skipna = 'True')
X['thalach'].hist()

X['chol'].max()
X['chol'].min()
X['chol'].mode()

X['thalach'].max()
X['thalach'].min()
X['thalach'].mode()

for i in X['chol']:
    if(i<200):
        X['chol']=X['chol'].replace(i,'Good')
    elif(i>=200 and i<=239):
        X['chol']=X['chol'].replace(i,'Borderline')
    else:
        X['chol']=X['chol'].replace(i,'High')

X.info()

X= pd.get_dummies(X,drop_first = 'True')

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state = 0)

from xgboost import XGBClassifier
Classifier = XGBClassifier(learning_rate =0.2,
 n_estimators=170, max_depth=2)

Classifier.fit(X_train , y_train)

import keras
from keras.models import Sequential #for initialsing our ann
from keras.layers import Dense #for creating layers in our ann

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = 15))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = Classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

rmse1=np.sqrt(np.mean((y_test-y_pred)**2))

 from sklearn.metrics import f1_score
f1_score(y_test , y_pred)







































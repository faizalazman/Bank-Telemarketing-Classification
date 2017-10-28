# Artificial Neural Network
# Result
# f1 score = 71.22
# precision score = 80.84
# recall = 63.65
    
# Part 1 - Data Preprocessing and EDA
    
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter   
from imblearn.datasets import make_imbalance     

# Importing the dataset
dataset = pd.read_csv('bank-additional-full.csv', sep = ";")
X_1 = dataset.iloc[:,:]
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1:].values
y = y.ravel()

###############################################################################
# EDA Section

# Check for missing values in all column

X_1.isnull().any().any()

# Datatype check

Counter(X.dtypes.values)
X_float = X_1.select_dtypes(include=['float64'])
X_int = X_1.select_dtypes(include=['int64'])

# Correlation plot for float

colormap = plt.cm.afmhot
plt.figure(figsize=(16,12))
plt.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(X_float.corr(),
            linewidths=0.1,
            vmax=1.0, square=True, 
            cmap=colormap, 
            linecolor='white', 
            annot=True)

# Correlation plot for int

colormap = plt.cm.afmhot
plt.figure(figsize=(16,12))
plt.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(X_float.corr(),
            linewidths=0.1,
            vmax=1.0, square=True, 
            cmap=colormap, 
            linecolor='white', 
            annot=True)

###############################################################################

# Limiting to only categorical object

X = X.select_dtypes(include=[object])

# one hot encoding on all the categories variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X = X.apply(le.fit_transform)

enc = OneHotEncoder()
X = enc.fit_transform(X).toarray()

# Label encoding for y
le_y = LabelEncoder()
y = le_y.fit_transform(y)
       
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Making imbalanced

X, y = make_imbalance(X, y,ratio = {0:4640, 1:4640})
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.15)

# Splitting the dataset into the Training set and validation set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.15)

###############################################################################
    
# Part 2 - Now let's make the ANN!
    
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
    
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 53, kernel_initializer = 'uniform', activation = 'relu', input_dim = 53))
    
# Adding the second hidden layer
classifier.add(Dense(units = 120, kernel_initializer = 'uniform', activation = 'relu'))
    
# Adding the Second layer
classifier.add(Dense(units = 120, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
    
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
# Fitting the ANN to the Training set
classifier.fit(X, y, batch_size = 1000, epochs = 60)

# Part 3 - Making predictions and evaluating the model
    
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

   
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, recall_score, precision_score,f1_score
cm = confusion_matrix(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
f1 = f1_score(y_test,y_pred)

###############################################################################

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
        classifier = Sequential()
        classifier.add(Dense(units = 53, kernel_initializer = 'uniform', activation = 'relu', input_dim = 53))
        classifier.add(Dense(units = 120, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = 120, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [200, 400,600,800,1000],
                  'epochs': [20,40,60,80,100,120,140,160,180,200],
              'optimizer': ['adam', 'rmsprop']}
grid_search = RandomizedSearchCV(estimator = classifier,
                               param_distributions = parameters,
                               scoring = 'accuracy',
                               cv = 5)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

###############################################################################

# Saving model architecture and weights

from keras.models import load_model
classifier.save('Hr_clf.h5')

# Load model architecture and weights

classifier = load_model('Hr_clf.h5')
     
###############################################################################    
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
        classifier = Sequential()
        classifier.add(Dense(units = 53, kernel_initializer = 'uniform', activation = 'relu', input_dim = 53))
        classifier.add(Dense(units = 120, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = 120, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.compile( optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])        
        return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 400, epochs = 200)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()


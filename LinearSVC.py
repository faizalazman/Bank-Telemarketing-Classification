# Linear SVC Classifier

# Result
# Recall = 59.20
# f1 Score = 67.38
# precision score = 78.18
    
# Part 1 - Data Preprocessing and EDA
    
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.svm import LinearSVC
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

# Part 2 Creating random forest model

svc_clf = LinearSVC(C = 0.01, max_iter = 100)
svc_clf.fit(X_train, y_train)

# Part 3 Prediction

y_pred = svc_clf.predict(X_test)

# Part 4 Evaluation 

from sklearn.metrics import confusion_matrix, recall_score, precision_score,f1_score
cm = confusion_matrix(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
f1 = f1_score(y_test,y_pred)

# Part 5 Evaluating using K-fold

from sklearn.model_selection import cross_val_score
CVS = cross_val_score(svc_clf,X_train,y_train, cv = 5, scoring = "roc_auc")
CVS = np.mean(CVS)
print(CVS)

# Part 5 Tuning the hyperparameters

from sklearn.model_selection import GridSearchCV
parameters = {'C': [0.01,0.1,1,10],
              'max_iter':[100,1000,10000]}
Grid_search= GridSearchCV( estimator = svc_clf, param_grid = parameters,scoring = 'roc_auc', cv = 5,verbose = 2)
Grid_search = Grid_search.fit(X_train, y_train)
best_parameters = Grid_search.best_params_


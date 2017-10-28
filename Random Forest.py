# Random Forest

# Result
# Recall = 60.20
# f1 Score = 64.76
# precision score = 70.06
    
# Part 1 - Data Preprocessing and EDA
    
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier 
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

rf_clf = RandomForestClassifier(n_estimators = 600, max_depth = 20, class_weight = 'balanced')
rf_clf.fit(X_train, y_train)

# Part 3 Prediction

y_pred = rf_clf.predict(X_test)

# Part 4 Evaluation

from sklearn.metrics import confusion_matrix, recall_score, precision_score,f1_score
cm = confusion_matrix(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
f1 = f1_score(y_test,y_pred)

# Part 5 Evaluating using K-fold

from sklearn.model_selection import cross_val_score
CVS = cross_val_score(rf_clf,X_train,y_train, cv = 5, scoring = "accuracy")
CVS = np.mean(CVS)
print(CVS)

# Part 5 Tuning the hyperparameters

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': [600],
              'max_depth': [20,22]}
Grid_search= GridSearchCV( estimator = rf_clf, param_grid = parameters,scoring = 'accuracy', cv = 5,verbose = 2)
Grid_search = Grid_search.fit(X_train, y_train)
best_parameters = Grid_search.best_params_
best_accuracy = Grid_search.best_score_


# ExtraTreesClassifier

# Result
# Recall = 63.51 
# f1 Score = 69.82
# precision score = 77.54
    
# Part 1 - Data Preprocessing and EDA
    
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.ensemble import ExtraTreesClassifier
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
plt.title('Pearson correlation of discrete features', y=1.05, size=15)
sns.heatmap(X_float.corr(),
            linewidths=0.1,
            vmax=1.0, square=True, 
            cmap=colormap, 
            linecolor='white', 
            annot=True)

# Ploting histogram for discrete data

his = X_int.loc[:, "age"]
plt.hist(his)
plt.title("Discrete Histogram")
plt.xlabel("age")
plt.ylabel("Frequency")

his = X_int.loc[:, "duration"]
plt.hist(his)
plt.title("Discrete Histogram")
plt.xlabel("duration")
plt.ylabel("Frequency")



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

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.98)
X_decom = pca.fit_transform(X)

# Making imbalanced

X, y = make_imbalance(X, y,ratio = {0:4640, 1:4640})
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.15)

# Part 2 Creating random forest model

et_clf = ExtraTreesClassifier(bootstrap =True, max_depth = 9, min_samples_split = 9, n_estimators = 100)
et_clf.fit(X_train, y_train)

# Part 3 Prediction

y_pred = et_clf.predict(X_test)

# Part 4 Evaluation

from sklearn.metrics import confusion_matrix, recall_score, precision_score,f1_score
cm = confusion_matrix(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
f1 = f1_score(y_test,y_pred)

# Part 5 Evaluating using K-fold

from sklearn.model_selection import cross_val_score
CVS = cross_val_score(et_clf,X_train,y_train, cv = 5, scoring = "accuracy")
CVS = np.mean(CVS)
print(CVS)

# Part 5 Tuning the hyperparameters

from sklearn.model_selection import GridSearchCV
parameters = {'bootstrap': [True, False],
              'max_depth': [3,6,9,12],
              'min_samples_split':[6,9,12],
              'n_estimators':[10,50,100]}
Grid_search= GridSearchCV( estimator = et_clf, param_grid = parameters,scoring = 'accuracy', cv = 5,verbose = 2)
Grid_search = Grid_search.fit(X_train, y_train)
best_parameters = Grid_search.best_params_
best_accuracy = Grid_search.best_score_

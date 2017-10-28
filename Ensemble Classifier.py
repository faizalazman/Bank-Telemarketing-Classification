# Ensemble Classifier

# Result
# Recall = 64.51
# f1 Score = 68.70
# precision score = 73.47
# Accuracy  = 90.05%
# Part 1 - Data Preprocessing and EDA
    
# Importing the libraries

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier 
from imblearn.datasets import make_imbalance 
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
        
# Importing the dataset

dataset = pd.read_csv('bank-additional-full.csv', sep = ";")
X_1 = dataset.iloc[:,:]
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1:].values
y = y.ravel()

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

# Classify using ensemble

et_clf = ExtraTreesClassifier(bootstrap =True, max_depth = 9, min_samples_split = 9, n_estimators = 100)
rf_clf = RandomForestClassifier(n_estimators = 600, max_depth = 20, class_weight = 'balanced')
xgb_clf = XGBClassifier(n_estimators = 100,learning_rate =0.1, max_depth = 3,min_child_weight = 1)
voting_clf = VotingClassifier(estimators = [('et',et_clf), ('rf', rf_clf),('xgb', xgb_clf)], voting ='hard')
voting_clf.fit(X_train,y_train)

# Part 3 Prediction

y_pred = voting_clf.predict(X_test)

# Part 4 Evaluation 

from sklearn.metrics import confusion_matrix, recall_score, precision_score,f1_score,accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
f1 = f1_score(y_test,y_pred)

# Part 5 Evaluating using K-fold

from sklearn.model_selection import cross_val_score
CVS = cross_val_score(voting_clf,X_train,y_train, cv = 5, scoring = "accuracy")
CVS = np.mean(CVS)
print(CVS)




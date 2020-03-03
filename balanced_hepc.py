# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:38:58 2020

@author: addod
"""

# importing the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report

data =pd.read_csv('HSV.csv')
# list(data) or 
list(data.columns) 
shape=data.shape
print(shape)
types = data.dtypes
print(types)
class_counts = data.groupby('staging').size()
print(class_counts)

def stage_type(series):
    if series == 1:
        return 0
    elif 1<= series < 6:
        return 1


data['stage_type'] = data['staging'].apply(stage_type)

data['stage_type'].value_counts(sort=False)
print(data.head)
data.drop("staging", axis=1, inplace=True) 
shape=data.shape
print(shape)
types = data.dtypes
print(types)
 

X_t=data.drop(['stage_type'],axis=1)
y_t=data['stage_type']


#----------------------------------------------------
#                    SMOTE METHOD
#----from imblearn.over_sampling import SMOTE
(np.random.seed(1234))
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_smote, y_smote = sm.fit_sample(X_t, y_t)
#------------------------------------

#from collections import Counter
#rint(sorted(Counter(y_russ).items()))
#X_rus.shape,y_russ.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_s= sc.fit_transform(X_smote)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 15)
X_pca_s= pca.fit_transform(X_s)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_ts, X_tes, y_ts, y_tes = train_test_split(X_pca_s, y_smote, test_size = 0.2, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
class_s = LogisticRegression(random_state = 0)

class_s.fit(X_ts, y_ts)

# Predicting the Test set results
y_pred = class_s.predict(X_tes)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_tes, y_pred)
print(classification_report(y_tes, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_tes,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_tes,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_tes,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_tes,y_pred)

# Fitting SVM to the Training set
#-------------------------------------------------
(np.random.seed(1234))
from sklearn.svm import SVC
from sklearn import svm
class_s=svm.SVC()
class_s.fit(X_ts, y_ts)

# Predicting the Test set results
y_pred = class_s.predict(X_tes)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_tes, y_pred)
print(classification_report(y_tes, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_tes,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_tes,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_tes,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_tes,y_pred)


#----------------------------------------------
#            RandomForest
#-----------------------------------------------
(np.random.seed(1234))
from sklearn.ensemble import RandomForestClassifier
reg=RandomForestClassifier()
reg.fit(X_ts, y_ts)
# Predicting the Test set results
y_pred = reg.predict(X_tes)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(y_tes, y_pred)
print(classification_report(y_tes, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_tes,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_tes,y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_tes,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_tes,y_pred)

#Fitting XGboost to the Training set
#-----------------------------------------------
(np.random.seed(1234))
from xgboost import XGBClassifier
class_s=XGBClassifier()
class_s.fit(X_ts, y_ts)

# Predicting the Test set results
y_pred = class_s.predict(X_tes)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_tes, y_pred)
print(classification_report(y_tes, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_tes,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_tes,y_pred)
print(recall)




#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_tes,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_tes,y_pred)
#----------------------------------------------------
#                    ADASYN METHOD
#---------------------------------------------------
(np.random.seed(1234))
from imblearn.over_sampling import ADASYN 
sa = ADASYN()
X_ad, y_ad = sa.fit_sample(X_t, y_t)
#from collections import Counter
#rint(sorted(Counter(y_russ).items()))
#X_rus.shape,y_russ.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler
ad = StandardScaler()
X_ads= ad.fit_transform(X_ad)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 15)
X_pca_ad= pca.fit_transform(X_ads)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_tad, X_tead, y_tad, y_tead = train_test_split(X_pca_ad, y_ad, test_size = 0.2, random_state = 0)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
class_ad = LogisticRegression(random_state = 0)

class_ad.fit(X_tad, y_tad)

# Predicting the Test set results
y_pred = class_ad.predict(X_tead)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_tead, y_pred)
print(classification_report(y_tead, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_tead,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_tead,y_pred)
print(recall)




#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_tead,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_tead,y_pred)

# Fitting SVM to the Training set
#-------------------------------------------------
(np.random.seed(1234))
from sklearn.svm import SVC
from sklearn import svm
class_ad=svm.SVC()
class_ad.fit(X_tad, y_tad)

# Predicting the Test set results
y_pred = class_ad.predict(X_tead)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_tead, y_pred)
print(classification_report(y_tead, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_tead,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_tead,y_pred)
print(recall)


#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_tead,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_tead,y_pred)


# Fitting RandomForestto the Training set
#------------------------------------------------
(np.random.seed(1234))
from sklearn.ensemble import RandomForestClassifier
class_ad=RandomForestClassifier()
class_ad.fit(X_tad, y_tad)

# Predicting the Test set results
y_pred = class_ad.predict(X_tead)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_tead, y_pred)
print(classification_report(y_tead, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_tead,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_tead,y_pred)
print(recall)


#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_tead,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_tead,y_pred)

#Fitting XGboost to the Training set
#-----------------------------------------------
(np.random.seed(1234))
from xgboost import XGBClassifier
class_ad=XGBClassifier()
class_ad.fit(X_tad, y_tad)

# Predicting the Test set results
y_pred = class_ad.predict(X_tead)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_tead, y_pred)
print(classification_report(y_tead, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_tead,y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_tead,y_pred)
print(recall)


#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_tead,y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_tead,y_pred)


#----------------------------------------------------
# LR: UNBALANCED DATASET, accuracy is 0.74 
#          0--0 and 1--1
#-----------------------------------------------------

#----------------------------------------------------
#                    SUMMARY OUTPUT LOGISTIC WITH
#                        BALANCED DATA
#---------------------------------------------------

# for smote, 
#accuracy is 0.66, 0--0.67,1--0.65

#for adasyn
#accuracy is 0.68, 0--0.71,1--0.66


#----------------------------------------------------
# SVM: UNBALANCED DATASET, accuracy is 0.74 
#          0--0 and 1--1
#-----------------------------------------------------

#----------------------------------------------------
#                    SUMMARY OUTPUT SVM WITH
#                        BALANCED DATA
#---------------------------------------------------

# for smote, 
#accuracy is 0.71, 0--0.66,1--0.77

#for adasyn
#accuracy is 0.63, 0--0.72,1--0.53

#-----------------------------------------------------------

#----------------------------------------------------
# DT: UNBALANCED DATASET, accuracy is 0.60
#          0--0.2 and 1--0.74
#-----------------------------------------------------

#----------------------------------------------------
#                    SUMMARY OUTPUT DT WITH
#                        BALANCED DATA
#---------------------------------------------------

# for smote, 
#accuracy is 0.64, 0--0.68,1--0.60

#for adasyn
#accuracy is 0.63, 0--0.68,1--0.58


#----------------------------------------------------


#----------------------------------------------------
# XG: UNBALANCED DATASET, accuracy is 0.75
#          0--0.06 and 1--1
#-----------------------------------------------------

#----------------------------------------------------
#                    SUMMARY OUTPUT DT WITH
#                        BALANCED DATA
#---------------------------------------------------

# for smote, 
#accuracy is 0.71, 0--0.67,1--0.75

#for adasyn
#accuracy is 0.7, 0--0.65,1--0.75

#----------------------------------------------------

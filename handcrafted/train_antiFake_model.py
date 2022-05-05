#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import csv
import os

#absolute path & before excuting the script, I deleted all the unnecessary columns except "isFake" and 26 features
# inputFile='D:/Users/Desktop/antiFake_result_origion_real_bseattel.csv'
inputFile = 'feat_data_10_10_80.csv'
df=pd.read_csv(inputFile)
data=df.values
data = data[:,3:]
data[:,1:26]=(data[:,1:26]-np.min(data[:,1:26],axis=0))/(np.max(data[:,1:26],axis=0)-np.min(data[:,1:26],axis=0))
np.random.seed(0)
np.random.shuffle(data)

#all
# spatial=data[:,15:22]
# histogram=data[:,1:14]
# frequency=data[:,23:26]

#additional
spatial=data[:,(15,16,19,21,22)]
histogram=data[:,(1,3,4,5,6,7,8,9,11,12,14)]
frequency=data[:,23:]

#strict salient
# spatial=data[:,(15,19,22)]
# histogram=data[:,(1,3,6,9,11,12)]
# frequency=data[:,(23,24,26)]


#normal salient
# spatial=data[:,15:22]
# histogram=data[:,1:14]
# frequency=data[:,(23,24,26)]

clf = svm.SVC(kernel='linear', C=1)
calculist=[]
calculist.append(spatial)
calculist.append(histogram)
calculist.append(frequency)
calculist.append(np.hstack((spatial,histogram)))
calculist.append(np.hstack((spatial,frequency)))
calculist.append(np.hstack((histogram,frequency)))
calculist.append(np.hstack((spatial,histogram,frequency)))

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

os.makedirs("records/zhao", exist_ok=True)

for test_size in [0.8]:
    with open('records/zhao/records' + str(int(test_size*100)) + '.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['accuracy', 'f1_score', 'precision_score', 'recall_score'])
    for calcu in calculist:
        X_train, X_test, y_train, y_test = train_test_split(calcu, data[:,0], test_size=test_size, random_state=0)
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,scoring='f1')
        clf.fit(X_train,y_train)
        best_parameters=clf.best_params_
        svm = SVC(**best_parameters)
        svm.fit(X_train, y_train)
        y_true,y_pred=y_test,svm.predict(X_test)
        # print(classification_report(y_true, y_pred))

        print([f1_score(y_true, y_pred),  precision_score(y_true, y_pred), recall_score(y_true, y_pred)])
        with open('records/zhao/records' + str(int(test_size*100)) + '.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([accuracy_score(y_true, y_pred), f1_score(y_true, y_pred),  precision_score(y_true, y_pred), recall_score(y_true, y_pred)])
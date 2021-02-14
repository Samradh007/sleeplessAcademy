import joblib
import math
import numpy as np 
import os 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

seed = 1
np.random.seed = seed

data_csv = 'C:\\Users\Abhishek\Desktop\proj\hack21\data\data_new.csv'
df = pd.read_csv(data_csv)
df = df.drop(['Frame_Num'], axis=1)
df = df.drop(df[df['Label'] == 'distracted'].index)
df['Label'] = df['Label'].astype('category').cat.codes

pnum = max(df['P_ID']) + 1
test_df, train_df = [row for _, row in df.groupby(df['P_ID'] <= 7)]

train_pnums = train_df['P_ID']
train_label = train_df['Label']
train_df = train_df.drop(['P_ID', 'Label'], axis=1)

test_pnums = test_df['P_ID']
test_label = test_df['Label']
test_df = test_df.drop(['P_ID', 'Label'], axis=1)

print (train_df.shape, test_df.shape)

print (metrics.confusion_matrix(test_label, test_label))

clf = RandomForestClassifier(n_estimators=100, random_state=seed)
clf.fit(train_df, train_label)
test_pred = clf.predict(test_df)

print ('accuracy', metrics.accuracy_score(test_label, test_pred))
print (metrics.confusion_matrix(test_label, test_pred))
print (metrics.classification_report(test_label, test_pred))

feature_imp = pd.Series(clf.feature_importances_, index = train_df.columns).sort_values(ascending = False) 
print (feature_imp)


model_path = 'C:\\Users\Abhishek\Desktop\proj\hack21\models_new\RF_drowsy_alert.joblib'
joblib.dump(clf, model_path)
print ('model saved')



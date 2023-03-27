# 구글 드라이브 연결
from google.colab import drive
drive.mount('/content/gdrive/')

# 구글 드라이브 경로 설정 
DATA_PATH = '/content/gdrive/My Drive/LG Aimers phase3/' 
MODEL_PATH='/content/gdrive/My Drive/LG Aimers phase3/'
SUBMISSION_PATH='/content/gdrive/My Drive/LG Aimers phase3/'

import pandas as pd
import random
import os
import numpy as np
import sklearn.metrics as metrics
import statistics
import math as math
import seaborn as sns

from sklearn.ensemble import VotingClassifier
from lightgbm import plot_importance
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(37) # Seed 고정

"""## Data Load"""

train_df = pd.read_csv(DATA_PATH + 'train.csv')
test_df = pd.read_csv(DATA_PATH + 'test.csv')

"""# EDA"""

# 칼럼 드랍
train_x = train_df.drop(columns=['PRODUCT_ID', 'Y_Class', 'Y_Quality'])
train_y = train_df['Y_Class']
test_x = test_df.drop(columns=['PRODUCT_ID'])

# train 데이터 확인 (rows 1132, columns 3328, 많은 결측치 존재)
print(train_x)

# Y_Class 분포 확인, class_imbalance => over_sampling 시도해보았으나 성능향상 x
sns.countplot(data=train_df, x='Y_Class')

"""#Data preprocessing"""

# 레이블인코딩
qual_col = ['LINE','PRODUCT_CODE']

for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train_x[i])
    train_x[i] = le.transform(train_x[i])
    
    for label in np.unique(test_x[i]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    test_x[i] = le.transform(test_x[i])
print('Done.')

# 결측치 처리
train_x = train_x.fillna(0)
test_x = test_x.fillna(0)

# 결측치 확인
print(train_x.isna().any())
print(test_x.isna().any())

"""# Data Split"""

from sklearn.model_selection import train_test_split

# train-test분리
X_train, X_test, y_train, y_test = train_test_split(train_x ,train_y, test_size=0.2)

# train-validation분리
X2_train, X_val, y2_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

"""# XGBClassifier, LGBMClassifier HyperParameter Tuning & Test"""

XGB = XGBClassifier(n_estimators=100, learning_rate=0.08, gamma = 0, subsample=0.75, colsample_bytree = 1, max_depth=7, alpha=4, n_jobs=-1
                   , booster='gbtree', importance_type='gain', min_child_weight=5
                   )
LGBM = LGBMClassifier(n_estimators=100, learning_rate=0.08, subsample=0.75, colsample_bytree = 1, max_depth=7, alpha=4, n_jobs=-1
                   , importance_type='gain', min_child_weight=5
                   )

XGB.fit(X_train, y_train)
preds = XGB.predict(X_test)

print("훈련 세트 정확도: {:.3f}".format(XGB.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(XGB.score(X_test, y_test)))
print("현재 Macro F1 Score의 검증 점수: {:.3f}".format(f1_score(preds, y_test, average = "macro")))
print("\n")  

XGB.fit(X2_train, y2_train)
preds2 = XGB.predict(X_val)

print("검증훈련 세트 정확도: {:.3f}".format(XGB.score(X2_train, y2_train)))
print("검증테스트 세트 정확도: {:.3f}".format(XGB.score(X_val, y_val)))
print("현재 Macro F1 Score의 검증 점수: {:.3f}".format(f1_score(preds2, y_val, average = "macro")))
print("\n") 

LGBM.fit(X_train, y_train)
preds = LGBM.predict(X_test) 

print("훈련 세트 정확도: {:.3f}".format(LGBM.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(LGBM.score(X_test, y_test)))
print("현재 Macro F1 Score의 검증 점수: {:.3f}".format(f1_score(preds, y_test, average = "macro")))
print("\n") 

LGBM.fit(X2_train, y2_train)
preds2 = LGBM.predict(X_val)

print("검증훈련 세트 정확도: {:.3f}".format(LGBM.score(X2_train, y2_train)))
print("검증테스트 세트 정확도: {:.3f}".format(LGBM.score(X_val, y_val)))
print("현재 Macro F1 Score의 검증 점수: {:.3f}".format(f1_score(preds2, y_val, average = "macro")))

"""#Voting Classifier Model"""

VM = VotingClassifier(
    estimators=[('XGB',XGB),('LGBM',LGBM)], 
    voting='hard', n_jobs=-1
    )

"""## Submit"""

VM.fit(train_x, train_y)
preds = VM.predict(test_x)

submit = pd.read_csv(DATA_PATH + './sample_submission.csv')

submit['Y_Class'] = preds

submit.to_csv(DATA_PATH + './code.csv', index=False)

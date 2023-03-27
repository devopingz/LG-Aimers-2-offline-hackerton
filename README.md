# LG-Aimers-2-offline-hackerton
스마트 공장의 제어 시스템 구축을 위한 제품 품질 분류 AI 모델 개발 (X_Feature 추가)

---
layout: single
title:  "jupyter notebook 변환하기!"
categories: coding
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


## Import



```python
# 구글 드라이브 연결
from google.colab import drive
drive.mount('/content/gdrive/')
```

<pre>
Drive already mounted at /content/gdrive/; to attempt to forcibly remount, call drive.mount("/content/gdrive/", force_remount=True).
</pre>

```python
# 구글 드라이브 경로 설정 
DATA_PATH = '/content/gdrive/My Drive/LG Aimers phase3/' 
MODEL_PATH='/content/gdrive/My Drive/LG Aimers phase3/'
SUBMISSION_PATH='/content/gdrive/My Drive/LG Aimers phase3/' 
```




```python
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
```




```python
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(37) # Seed 고정
```

## Data Load



```python
train_df = pd.read_csv(DATA_PATH + 'train.csv')
test_df = pd.read_csv(DATA_PATH + 'test.csv')
```

# EDA



```python
# 칼럼 드랍
train_x = train_df.drop(columns=['PRODUCT_ID', 'Y_Class', 'Y_Quality'])
train_y = train_df['Y_Class']
test_x = test_df.drop(columns=['PRODUCT_ID'])
```


```python
# train 데이터 확인 (rows 1132, columns 3328, 많은 결측치 존재)
print(train_x)
```

<pre>
         LINE PRODUCT_CODE  X_1    X_2  X_3   X_4   X_5  X_6   X_7   X_8  ...  \
0     T100304         T_31  2.0   95.0  0.0  45.0  10.0  0.0  45.0  10.0  ...   
1     T100306         T_31  2.0   96.0  0.0  45.0  10.0  0.0  53.0  10.0  ...   
2     T100306         T_31  2.0   95.0  0.0  45.0  10.0  0.0  60.0  10.0  ...   
3     T100306         T_31  2.0   87.0  0.0  45.0  10.0  0.0  53.0  10.0  ...   
4     T100306         T_31  2.0   95.0  0.0  45.0  10.0  0.0  51.0  10.0  ...   
...       ...          ...  ...    ...  ...   ...   ...  ...   ...   ...  ...   
1127  T050304         A_31  NaN    NaN  NaN   NaN   NaN  NaN   NaN   NaN  ...   
1128  T100304         T_31  2.0  102.0  0.0  45.0  11.0  0.0  45.0  10.0  ...   
1129  T100306         T_31  1.0   88.0  0.0  45.0  10.0  0.0  51.0  10.0  ...   
1130  T010306         A_31  NaN    NaN  NaN   NaN   NaN  NaN   NaN   NaN  ...   
1131  T100304         T_31  2.0  101.0  0.0  45.0  11.0  0.0  45.0  10.0  ...   

        X_3317    X_3318    X_3319    X_3320    X_3321    X_3322    X_3323  \
0     0.000008  0.000003  0.191408  0.000008  0.001210  0.000021  0.000003   
1     0.000008  0.000003  0.188993  0.000032  0.000644  0.000041  0.000002   
2          NaN       NaN       NaN       NaN       NaN       NaN       NaN   
3     0.000007  0.000003  0.189424  0.000034  0.000678  0.000043  0.000004   
4          NaN       NaN       NaN       NaN       NaN       NaN       NaN   
...        ...       ...       ...       ...       ...       ...       ...   
1127       NaN       NaN       NaN       NaN       NaN       NaN       NaN   
1128       NaN       NaN       NaN       NaN       NaN       NaN       NaN   
1129       NaN       NaN       NaN       NaN       NaN       NaN       NaN   
1130       NaN       NaN       NaN       NaN       NaN       NaN       NaN   
1131  0.000006  0.000004  0.190968  0.000009  0.001270  0.000022  0.000004   

        X_3324  X_3325    X_3326  
0     0.000002   0.189  0.000006  
1     0.000003   0.185  0.000029  
2          NaN     NaN       NaN  
3     0.000003   0.188  0.000031  
4          NaN     NaN       NaN  
...        ...     ...       ...  
1127       NaN     NaN       NaN  
1128       NaN     NaN       NaN  
1129       NaN     NaN       NaN  
1130       NaN     NaN       NaN  
1131  0.000001   0.190  0.000005  

[1132 rows x 3328 columns]
</pre>

```python
# Y_Class 분포 확인, class_imbalance => over_sampling 시도해보았으나 성능향상 x
sns.countplot(data=train_df, x='Y_Class')
```

<pre>
<Axes: xlabel='Y_Class', ylabel='count'>
</pre>
<pre>
<Figure size 432x288 with 1 Axes>
</pre>
#Data preprocessing



```python
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
```

<pre>
Done.
</pre>

```python
# 결측치 처리
train_x = train_x.fillna(0)
test_x = test_x.fillna(0)
```


```python
# 결측치 확인
print(train_x.isna().any())
print(test_x.isna().any())
```

<pre>
LINE            False
PRODUCT_CODE    False
X_1             False
X_2             False
X_3             False
                ...  
X_3322          False
X_3323          False
X_3324          False
X_3325          False
X_3326          False
Length: 3328, dtype: bool
LINE            False
PRODUCT_CODE    False
X_1             False
X_2             False
X_3             False
                ...  
X_3322          False
X_3323          False
X_3324          False
X_3325          False
X_3326          False
Length: 3328, dtype: bool
</pre>
# Data Split



```python
from sklearn.model_selection import train_test_split

# train-test분리
X_train, X_test, y_train, y_test = train_test_split(train_x ,train_y, test_size=0.2)

# train-validation분리
X2_train, X_val, y2_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
```

# XGBClassifier, LGBMClassifier HyperParameter Tuning & Test



```python
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
```

<pre>
훈련 세트 정확도: 0.937
테스트 세트 정확도: 0.722
현재 Macro F1 Score의 검증 점수: 0.560


검증훈련 세트 정확도: 0.934
검증테스트 세트 정확도: 0.718
현재 Macro F1 Score의 검증 점수: 0.579


훈련 세트 정확도: 0.997
테스트 세트 정확도: 0.780
현재 Macro F1 Score의 검증 점수: 0.666


검증훈련 세트 정확도: 1.000
검증테스트 세트 정확도: 0.713
현재 Macro F1 Score의 검증 점수: 0.570
</pre>
#Voting Classifier Model



```python
VM = VotingClassifier(
    estimators=[('XGB',XGB),('LGBM',LGBM)], 
    voting='hard', n_jobs=-1
    )
```

## Submit



```python
VM.fit(train_x, train_y)
preds = VM.predict(test_x)
```


```python
submit = pd.read_csv(DATA_PATH + './sample_submission.csv')
```


```python
submit['Y_Class'] = preds
```


```python
submit.to_csv(DATA_PATH + './code.csv', index=False)
```

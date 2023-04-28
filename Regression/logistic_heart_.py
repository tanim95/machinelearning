# -*- coding: utf-8 -*-


# Original file is located at
#     https://colab.research.google.com/drive/1TducI_zch37ZAEEpB9LCuLT3NEKz-dq

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""## Data

This database contains 14 physical attributes based on physical testing of a patient. Blood samples are taken and the patient also conducts a brief exercise test. The "goal" field refers to the presence of heart disease in the patient. It is integer (0 for no presence, 1 for presence). In general, to confirm 100% if a patient has heart disease can be quite an invasive process, so if we can create a model that accurately predicts the likelihood of heart disease, we can help avoid expensive and invasive procedures.

Content

Attribute Information:

* age
* sex
* chest pain type (4 values)
* resting blood pressure
* serum cholestoral in mg/dl
* fasting blood sugar > 120 mg/dl
* resting electrocardiographic results (values 0,1,2)
* maximum heart rate achieved
* exercise induced angina
* oldpeak = ST depression induced by exercise relative to rest
* the slope of the peak exercise ST segment
* number of major vessels (0-3) colored by flourosopy
* thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
* target:0 for no presence of heart disease, 1 for presence of heart disease

Original Source: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

Creators:

Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

----

**TASK: Run the cell below to read in the data.**
"""

df = pd.read_csv('heart.csv')

df.head()

df['target'].unique()

df.describe().transpose()

df.info()

sns.countplot(data=df, x='target')

plt.figure(figsize=(10, 10), dpi=100)
sns.heatmap(df.corr(), annot=True)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=20, random_state=101)


scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


model = LogisticRegressionCV(cv=10, solver='saga', penalty='l1', max_iter=500)

model.fit(scaled_X_train, y_train)

model.C_

model.get_params()

model.coef_

coef = pd.Series(index=X.columns, data=model.coef_[0])
sns.barplot(x=coef.index, y=coef.values)

y_pred = model.predict(scaled_X_test)
y_pred

"""---------

## Model Performance Evaluation
"""


accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred)

"""**Final Task: A patient with the following features has come into the medical office:**

    age          48.0
    sex           0.0
    cp            2.0
    trestbps    130.0
    chol        275.0
    fbs           0.0
    restecg       1.0
    thalach     139.0
    exang         0.0
    oldpeak       0.2
    slope         2.0
    ca            0.0
    thal          2.0

**TASK: What does your model predict for this patient? Do they have heart disease? How "sure" is your model of this prediction?**
"""

new_sample = [[48.0, 0.0, 2.0, 130.0, 275.0,
               0.0, 1.0, 139.0, 0.0, 0.2, 2.0, 0.0, 2.0]]
scaled_new_sample = scaler.transform(new_sample)

y_pred_new = model.predict(scaled_new_sample)

y_pred_new

prediction_probability = model.predict_log_proba(scaled_new_sample)
prediction_probability
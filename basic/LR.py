from sklearn.linear_model import LogisticRegression

import pandas as pd

import math


data = pd.read_table('/Users/meituan_sxw/Downloads/flower.txt',header=None)

data.columns = ['f1','f2','f3','f4','flower']

data['label'] = data['flower'].map(lambda x:0 if x=='setosa' else 1)

print(data)
x = data[['f1','f2','f3','f4']].values.tolist()
y = data[['label']]

lr = LogisticRegression(penalty="l2",fit_intercept=False)

lr.fit(x,y)

print(lr.predict_proba(x))

print(1 / (1 + math.pow(math.e,5.1*(-0.40247392)+3.5*(-1.46382925)+1.4*2.23785648+0.2 * 1.00009294+(-0.25906453))))


print(lr.coef_)

print(lr.__dict__)
from random import random
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_data():
    train_data = pd.read_table("data/adult.data.txt",header=None,delimiter=',')
    test_data = pd.read_table("data/adult.test.txt",header=None,delimiter=',')

    all_columns = ['age','workclass','fnlwgt','education','education-num',
                        'marital-status','occupation','relationship','race','sex',
                        'capital-gain','capital-loss','hours-per-week','native-country','label','type']

    continus_columns = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    dummy_columns = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']

    train_data['type'] = 1
    test_data['type'] = 2

    all_data = pd.concat([train_data,test_data],axis=0)
    all_data.columns = all_columns

    all_data = pd.get_dummies(all_data,columns=dummy_columns)


    test_data = all_data[all_data['type']==2].drop(['type'],axis=1)
    train_data = all_data[all_data['type']==1].drop(['type'],axis=1)

    train_data['label'] = train_data['label'].map(lambda x: 1 if x.strip() == '>50K' else 0)
    test_data['label'] = test_data['label'].map(lambda x: 1 if x.strip() == '>50K.' else 0)


    for col in continus_columns:
        ss = StandardScaler()
        train_data[col] = ss.fit_transform(train_data[[col]])
        test_data[col] = ss.transform(test_data[[col]])


    train_y = train_data['label']
    train_x = train_data.drop(['label'],axis=1)
    test_y = test_data['label']
    test_x = test_data.drop(['label'],axis=1)

    return train_x,train_y,test_x,test_y

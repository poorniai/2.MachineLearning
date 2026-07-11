import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_dummies():
    dataset = pd.read_csv("50_Startups.csv")
    dataset = pd.get_dummies(dataset, dtype=int, drop_first=True)
    X = dataset[['R&D Spend', 'Administration', 'Marketing Spend',
                 'State_Florida', 'State_New York']]
    y = dataset[['Profit']]
    return X, y

def split_train_test(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
    return X_train,X_test,y_train,y_test

def preprocess_scaler(X_train,X_test):
    sc = StandardScaler()
    X_train= sc.fit_transform(X_train)
    X_test= sc.transform(X_test)
    return X_train, X_test,sc


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



df = pd.read_csv('./data/data_bank.csv')
X = df.drop(['CreditCard'],axis=1).to_numpy()
y = df['CreditCard']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# without sklearn
class SVM:

    def __init__(self, learning_rate, lambda_param, n_iters):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
 
lr = [5e-3, 1e-3, 5e-4, 1e-4]
lambda_params = [0.01,0.03,0.05,0.1]
n_iters = [100, 200, 500, 1000, 2000]   
acc_lst = []
lr_lst= []
lam_lst = []
iters = []

def run_SVM():
    result = pd.DataFrame()
    for i in lr:
        for j in lambda_params:
            for k in n_iters:
                        
                clf = SVM(i,j,k)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred = np.where(y_pred <= -1, 0, 1)
                acc = accuracy_score(y_test,y_pred)
                
                acc_lst.append(acc)
                lr_lst.append(i)
                lam_lst.append(j)
                iters.append(k)
                
    result['learning rate'] = lr_lst
    result['lambda parameters'] = lam_lst
    result['n_iters'] = iters
    result['acc'] = acc_lst
                
    result.to_csv('SVM_result.csv',index=False)

#run_SVM()


# with sklearn 
    
from sklearn.svm import SVC

def run_clf(kernel_name):
    clf = SVC(kernel=kernel_name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = np.where(y_pred <= -1, 0, 1)
    acc = accuracy_score(y_test,y_pred)
    return acc      
        
def run_SVM_2():
    
    result = pd.DataFrame()
    acc_lst = []
    for i in ['linear','poly','rbf','sigmoid']:
        acc = run_clf(i)
        acc_lst.append(acc)
    
    result['kernel'] = ['linear','poly','rbf','sigmoid']
    result['acc'] = acc_lst 
    result.to_csv('SVM_result_sklearn.csv',index=False)
    
#run_SVM_2()

# classification for baseline 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)
from email.utils import parsedate_to_datetime
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./data/data_insurance.csv')
X = df.drop(['charges'],axis=1)
y = df['charges']

sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# with sklearn 
from sklearn.svm import SVR

def run_clf(kernel_name):
    clf = SVR(kernel=kernel_name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    r2 = clf.score(X_test,y_test)
    mse = mean_squared_error(y_pred, y_test)
    return r2, mse  
        
def run_SVR():
    
    result = pd.DataFrame()
    mse_lst = []
    r2_lst = []
    for i in ['linear','poly','rbf','sigmoid']:
        r2, mse = run_clf(i)
        mse_lst.append(mse)
        r2_lst.append(r2)
    
    result['kernel'] = ['linear','poly','rbf','sigmoid']
    result['mse'] = mse_lst
    result['r2'] = r2_lst
    result.to_csv('SVR_result_sklearn.csv',index=False)
    
#run_SVR()


# GridSearch 
# from sklearn.model_selection import GridSearchCV

# param_grid = [
#     {'kernel' : ['linear'], 'C': [10.,30.,100.,300.,1000.,3000.,10000.,30000.],
#      'gamma' : ['auto', 0.1, 0.01], 'epsilon' : [0.2, 0.1, 0.05, 0.02, 0.01]
#      }]

# svm_reg = SVR()

# grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring = 'r2', verbose=2)
# grid_search.fit(X_train, y_train)

# print(grid_search.best_params_) # {'C': 3000.0, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'linear'}

# svr_best = SVR(kernel='linear', C=3000, epsilon=0.01, gamma='auto')
# svr_best.fit(X_train,y_train)

# y_pred = svr_best.predict(X_test)
# r2 = svr_best.score(X_test,y_test)
# mse = mean_squared_error(y_pred, y_test)
# print(r2,mse) # 0.7060211957639762 48733657.34117352


# linear regression for baseline 
# from sklearn.linear_model import LinearRegression 
# linear = LinearRegression()

# linear.fit(X_train,y_train)
# y_pred = linear.predict(X_test)
# r2= linear.score(X_test,y_test)
# mse = mean_squared_error(y_pred, y_test)
# print(r2, mse) # 0.7944038492696281 34082227.07210547


# conduct SVR with generated non-linear data

# create new data
X = np.sort(10*np.random.rand(100,1), axis=0) 
y = np.sin(X).ravel()                         

y[::5] +=2*(0.5-np.random.rand(20))           
y[::4] +=3*(0.5-np.random.rand(25))
y[::1] +=1*(0.5-np.random.rand(100))

plt.plot(X,y)
plt.xlabel('X data')
plt.ylabel('y data')
plt.title('non-linear data')
plt.savefig('./pics/non-linear_data.png')

# SVR with non-linear data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
def run_SVR_1():
    
    result = pd.DataFrame()
    mse_lst = []
    r2_lst = []
    for i in ['linear','poly','rbf','sigmoid']:
        r2, mse = run_clf(i)
        mse_lst.append(mse)
        r2_lst.append(r2)
    
    result['kernel'] = ['linear','poly','rbf','sigmoid']
    result['mse'] = mse_lst
    result['r2'] = r2_lst
    result.to_csv('SVR_result_nonlinear.csv',index=False)
    
#run_SVR_1()

# GridSearch with RBF kernel 
from sklearn.model_selection import GridSearchCV

def grid2():
    param_grid = [
        {'kernel' : ['rbf'], 'C': [10.,30.,100.,300.,1000.,3000.,10000.,30000.],
        'gamma' : ['auto', 0.1, 0.01], 'epsilon' : [0.2, 0.1, 0.05, 0.02, 0.01]
        }]

    svm_reg = SVR()

    grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring = 'r2', verbose=2)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(grid_search.best_params_) # {'C': 10.0, 'epsilon': 0.2, 'gamma': 0.1, 'kernel': 'rbf'}

    svr_best = SVR(**best_params)
    svr_best.fit(X_train,y_train)

    y_pred = svr_best.predict(X_test)
    r2 = svr_best.score(X_test,y_test)
    mse = mean_squared_error(y_pred, y_test)
    print(r2,mse) # 0.5801512939031991 0.32302460305891

grid2()
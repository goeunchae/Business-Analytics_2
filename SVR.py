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



# without sklearn 
def kernel_f(xi, xj, kernel = None, coef0=1.0, degree=3, gamma=0.1):

    if kernel == 'linear':                                  # Linear kernel
        result = np.dot(xi,xj)+coef0
    elif kernel == 'poly':                                  # Polynomial kernel
        result = (np.dot(xi,xj)+coef0)**degree
    elif kernel == 'rbf':                                   # RBF kernel
        result = np.exp(-gamma*np.linalg.norm(xi-xj)**2)
    elif kernel =='sigmoid':                                # Sigmoid kernel
        result = np.tanh(gamma*np.dot(xi,xj)+coef0)
    else:                                                   # Dot product
        result = np.dot(xi,xj)

    return result

def kernel_matrix(X, kernel, coef0=1.0, degree=3, gamma=0.1):

    X = np.array(X,dtype=np.float64)
    mat = []
    for i in X:
        row = []
        for j in X:
            if kernel=='linear':           
                row.append(kernel_f(i, j, kernel = 'linear', coef0))   
            elif kernel=='poly':
                row.append(kernel_f(i, j, kernel = 'poly', coef0, degree))
            elif kernel=='rbf':
                row.append(kernel_f(i,j,kernel = 'rbf', gamma))
            elif kernel =='sigmoid':
                row.append(kernel_f(i,j,kernel = 'sigmoid', coef0, gamma))    
            else:
                row.append(np.dot(i,j))
        mat.append(row)

    return mat

## Epsilon - insensitive loss
def eps_loss(t, c=3, e = 5):
    return(abs(t)<e)*0* abs(t) +((t)>=e)*c*abs(t-e) +((t)<-e)*c*abs(t+e)

## Laplacian loss
def laplacian_loss(t, c=3):
    return c*abs(t)

## Gaussian loss
def gaussian_loss(t, c=3):
    return c*0.5*t**2   

## Hubor loss
def huber_loss(t, c=3, s=5):
    return c*((abs(t)<s)*(0.5/s)*(t**2) + (abs(t)>=s)*(abs(t)-s/2))

## Quadratic form
class svr:
    '''
    loss: ['epsilon-insensitive','huber','laplacian','gaussian','polynomial','piecewise_polynomial'] defualt: epsilon-insensitive
    kernel: [None,'rbf', 'linear','poly']
    coef0: Independent term in kernel function. in 'linear','poly'
    degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    gamma: Kernel coefficient for ‘rbf’, ‘poly’
    p: polynomial/'piecewise_polynomial loss function p
    sigma: for loss function
    '''
    def __init__(self,loss = 'epsilon-insensitive',C = 1.0, epsilon = 1.0,
                 kernel = None, coef0=1.0, degree=3, gamma=0.1, p=3, sigma=0.1):
        self.loss = loss
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.coef0 = coef0
        self.degree= degree
        self.gamma = gamma
        self.sigma = sigma
        self.p = p

    #model fitting    
    def fit(self,X,y):
        self.X = X
        self.y = y

        n = self.X.shape[0]#numper of instances
        #variable for dual optimization problem

        alpha = cvx.Variable(n)
        alpha_ = cvx.Variable(n)
        one_vec = np.ones(n)

        #object function and constraints of all types of loss fuction

        if self.loss == 'huber':
            self.constraints = []
            self.svr_obj = cvx.Maximize(-.5*cvx.quad_form(alpha-alpha_, kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) - self.epsilon*one_vec*(alpha+alpha_) + self.y*(alpha-alpha_) - self.sigma/(2*self.C)*one_vec*(cvx.power(alpha, 2) + cvx.power(alpha_, 2)))
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]
            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()

            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()

            #compute b
            idx = np.where((np.array(alpha.value).ravel() < self.C-1E-10) * (np.array(alpha.value).ravel() > 1E-10))[0][0]
            self.b = -self.epsilon + self.y[idx] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx], self.X[i],
                                       kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])

        elif self.loss == 'laplacian': # epsilon = 0 of epsilon-insensitive
            self.constraints = []
            self.svr_obj = cvx.Maximize(-.5*cvx.quad_form(alpha-alpha_,
                                                     kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) + self.y*(alpha-alpha_))
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]
            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0, alpha[i] <= self.C, alpha_[i] <= self.C]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()

            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()

            #compute b
            idx = np.where((np.array(alpha.value).ravel() < self.C-1E-10) * (np.array(alpha.value).ravel() > 1E-10))[0][0]
            self.b = -self.epsilon + self.y[idx] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx], self.X[i],
                                       kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])

        elif self.loss == 'gaussian': # sigma = 1 of huber
            self.constraints = []
            self.svr_obj = cvx.Maximize(-.5*cvx.quad_form(alpha-alpha_,
                                                     kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) - self.epsilon*one_vec*(alpha+alpha_) + self.y*(alpha-alpha_) - 1./(2*self.C)*one_vec*(cvx.power(alpha, 2) + cvx.power(alpha_, 2)))
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]
            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()

            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()

            #compute b
            idx = np.where((np.array(alpha.value).ravel() < self.C-1E-10) * (np.array(alpha.value).ravel() > 1E-10))[0][0]
            self.b = -self.epsilon + self.y[idx] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx], self.X[i],
                                       kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])

        elif self.loss == 'polynomial':
            self.constraints = []
            self.svr_obj = cvx.Maximize(-.5*cvx.quad_form(alpha-alpha_,
                                                     kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) + self.y*(alpha-alpha_) - (self.p-1)/(self.p*self.C**(self.p-1))* one_vec*(cvx.power(alpha, self.p/(self.p-1)) + cvx.power(alpha_, self.p/(self.p-1))))
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]
            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()

            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()

            #compute b
            idx = np.where((np.array(alpha.value).ravel() < self.C-1E-10) * (np.array(alpha.value).ravel() > 1E-10))[0][0]
            self.b = -self.epsilon + self.y[idx] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx], self.X[i],
                                       kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])

        elif self.loss == 'piecewise_polynomial':
            self.constraints = []
            self.svr_obj = cvx.Maximize(-.5*cvx.quad_form(alpha-alpha_,
                                                     kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) + self.y*(alpha-alpha_) - (self.p-1)*self.sigma/(self.p*self.C**(self.p-1))* one_vec*(cvx.power(alpha, self.p/(self.p-1)) + cvx.power(alpha_, self.p/(self.p-1))))
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]
            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()

            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()

            #compute b
            idx = np.where((np.array(alpha.value).ravel() < self.C-1E-10) * (np.array(alpha.value).ravel() > 1E-10))[0][0]
            self.b = -self.epsilon + self.y[idx] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx], self.X[i],
                                       kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
        else:
            self.constraints = []
            self.svr_obj = cvx.Maximize(-.5*cvx.quad_form(alpha-alpha_, kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) - self.epsilon*one_vec*(alpha+alpha_) + self.y*(alpha-alpha_))
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]
            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0, alpha[i] <= self.C, alpha_[i] <= self.C]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()

            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()

            #compute b
            idx = np.where((np.array(alpha.value).ravel() < self.C-1E-10) * (np.array(alpha.value).ravel() > 1E-10))[0][0]
            self.b = -self.epsilon + self.y[idx] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx], self.X[i],
                                       kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])


    def predict(self,new_X):
        self.results = []
        n = new_X.shape[0]
        for j in range(n):
            X = new_X[j]
            self.results += [np.sum([(self.a[i] - self.a_[i]) *kernel_f(X, self.X[i],
                                     kernel = self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)]) + self.b]   
        return self.results



# with sklearn 
from sklearn.svm import SVR

def run_clf(kernel_name):
    clf = SVR(kernel=kernel_name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    r2 = clf.score(X_test,y_test)
    mse = mean_squared_error(y_pred, y_test)
    return r2, mse  
        
def run_SVR_2():
    
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
    
run_SVR_2()

# linear regression for baseline 
from sklearn.linear_model import LinearRegression 
linear = LinearRegression()

linear.fit(X_train,y_train)
y_pred = linear.predict(X_test)
r2= linear.score(X_test,y_test)
mse = mean_squared_error(y_pred, y_test)
print(r2, mse)


# conduct SVR with generated non-linear data

# generate data 
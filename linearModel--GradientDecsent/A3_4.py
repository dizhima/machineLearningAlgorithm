#coding:UTF-8  

'''
refrence:
http://blog.csdn.net/google19890102/article/details/46389869
'''

import numpy as np
import pandas as pd
from function import *  
from numpy import *
import matplotlib.pyplot as plt


def lbfgs(fun, gfun, x0): 
    weights = []
    result = []#final result of func
    maxk = 30#max iteration num
    rho = 0.1  
    sigma = 0.4  
      
    H0 = eye(shape(x0)[0])  
      
    s = []  
    y = []  
    m = 6  
      
    k = 1  
    gk = mat(gfun(x0))  
    dk = -H0 * gk  
    while (k < maxk):
        n = 0  
        mk = 0  
        gk = mat(gfun(x0))
        while (n < 20):  
            newf = fun(x0 + rho ** n * dk)  
            oldf = fun(x0)  
            if (newf < oldf + sigma * (rho ** n) * (gk.T * dk)[0, 0]):  
                mk = n  
                break  
            n = n + 1  
          
        #LBFGS correction
        x = x0 + rho ** mk * dk  
        #print x  
          
        if k > m:  
            s.pop(0)  
            y.pop(0)  
              
        sk = x - x0  
        yk = gfun(x) - gk  
          
        s.append(sk)  
        y.append(yk)  
          
        #two-loop process 
        t = len(s)  
        qk = gfun(x)  
        a = []  
        for i in range(t):  
            alpha = (s[t - i - 1].T * qk) / (y[t - i - 1].T * s[t - i - 1])  
            qk = qk - alpha[0, 0] * y[t - i - 1]  
            a.append(alpha[0, 0])  
        r = H0 * qk  
              
        for i in range(t):  
            beta = (y[i].T * r) / (y[i].T * s[i])  
            r = r + s[i] * (a[t - i - 1] - beta[0, 0])  
  
              
        if (yk.T * sk > 0):  
            dk = -r              
          
        k = k + 1  
        x0 = x  
        weights.append(np.squeeze(np.asarray(x0)))
        result.append(fun(x0))  
      
    return result,weights


def get_data(data):
    filename = ''
    if data == 0:
        filename = 'A3.train.csv'
    elif data == 1:
        filename = 'A3.test.csv'
    else:
        print('get data error!')
        return
    df = pd.read_csv(filename)
    df = np.array(df)
    N = len(df)
    train_x = []
    train_y = []
    for example in df:
        train_y.append(example[0])
        train_x.append(example[1:])
    return [train_x,train_y]


def predict(x, w):
    '''
    compute the activation of the example
    :para np.array x, w
    :para float b
    :return float: activation
    '''
    xx = x.tolist() + [1.]
    xx = np.array(xx)
    a = np.dot(xx, w)
    return a


def grad(x, y, w, style):
    xx = x.tolist() + [1.]
    xx = np.array(xx)
    if style == 0:
        yhat = predict(x, w)
        dw = - 2 * (y - yhat) * xx
        
    if style == 1:
        yhat = predict(x, w)
        dw = - y / (1 + np.exp(y * yhat)) * xx
        
    return dw


def func(w):
    tempw = np.squeeze(np.asarray(w))
    N = len(train_x)
    loss = 0
    for i in range(N):
        xx = train_x[i]
        yhat = predict(xx, tempw)
        loss += np.log(1 + np.exp(train_y[i] * yhat))
    loss /= N
    return loss


def gfunc(w):
    tempw = np.squeeze(np.asarray(w))
    N = len(train_x)
    for i in range(N):
        gg = (np.zeros(len(train_x[0]) + 1))
        for i in range(len(train_x)):
            gg += (grad(train_x[i], train_y[i], tempw, 0))
    gg /= N
    return mat(gg).T


def test_accuracy(test, w):
    test_x = test[0]
    test_y = test[1]
    N = len(test_x)
    acc = 0
    for i in range(N):
        a = predict(test_x[i], w)
        if a * test_y[i] > 0:
            acc += 1
    return acc / N


def ploting(Y, filename, ylabel, title):
    '''
    plot a figure 
    '''
    X = list(range(1,len(Y)+1))
    X = np.array(X)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(X, Y, linewidth = '1',color='b')
    ax.set_xlabel('epochs')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(filename+'.png')
    return


if __name__ == "__main__":
    train = get_data(data = 0)
    train_x = train[0]
    train_y = train[1]
    w0 = mat(np.zeros(9)).T

    result, weights = lbfgs(func, gfunc, w0)
    test = get_data(data = 1)
    Acc = []
    for w in weights:
        Acc.append(test_accuracy(test, w))

    ploting(Acc,'acc',
            'accuracy',
           ' accuracy for LGBFS')

    ploting(result,'loss',
           'loss',
           'loss versus epoch for L-BFGS')

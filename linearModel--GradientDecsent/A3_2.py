import random
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt


SAMPLE = 100


def test_accuracy(test, w):
    '''
    compute the test data accuracy
    :para np.array test: test data
    :para np.array w: weights
    '''
    test_x = test[0]
    test_y = test[1]
    N = len(test_x)
    acc = 0
    for i in range(N):
        a = predict(test_x[i], w)
        if a * test_y[i] > 0:
            acc += 1
    return acc / N


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

def average_squared_error(train, w, losst = 1):
    '''
    compute the average quared error
    :para np.array train_x, train_y: training data 
    :para np.array w: weights
    :para float b: bias
    :return float: Average Squared Error
    '''
    train_x = train[0]
    train_y = train[1]
    N = len(train_x)
    error = 0
    for i in range(N):
        a = predict(train_x[i], w)
        if a <= 0:
            error += (-1 - train_y[i]) ** 2
        else:
            error += (1 - train_y[i]) ** 2
        
        
    error = error / N
    return error


def develop_data(train_x, train_y, dev_index):
    '''
    derive the develop data
    :para np.array train_x, train_y: training data 
    :para np.array dev_index: develop data index in train data
    :return list: develop data
    '''
    dev_x = []
    dev_y = []
    for i in dev_index:
        dev_x.append(train_x[i])
        dev_y.append(train_y[i])
        del train_x[i]
        del train_y[i]
    return [np.array(dev_x), np.array(dev_y)]


def gradient(x, y, w, style):
    xx = x.tolist() + [1.]
    xx = np.array(xx)
    if style == 0:
        yhat = predict(x, w)
        dw = 2 * (y - yhat) * xx
        
    if style == 1:
        yhat = predict(x, w)
        dw = y / (1 + np.exp(y * yhat)) * xx
        
    return dw


def update(x, y, w, stepSize, style = 1):
    '''
    updata the weights and bias
    :para np.array train_x[i]: example's feature values
    :para int train_y[i]:example's label
    :para np.array w: original weights
    :para float b: original bias
    :para int style: 0 for linear regression, 1 for logistic 
    :return :new weights and bias
    '''
    xx = x.tolist() + [1.]
    xx = np.array(xx)
    nw = np.array(w)
    if style ==0:
        if max(nw) > 1e300:
            nw /= max(nw)
        yhat = predict(x, nw)       
        nw += stepSize * 2 * (y - yhat) * xx
#         nb = 0
    
    if style == 1:
        yhat = predict(x, nw)
        nw += stepSize * y / (1 + np.exp(y * yhat)) * xx
        
    return nw


def stochastic_gradien_descent(train, stepSize, epochs = 10000, 
                              sample = SAMPLE, dev_size = 0.2
                               , losstype = 1,
                               best = 0, GD = False,
                              asc = False):
    
    train_x = train[0]
    train_y = train[1]
    # initiate z
    N = len(train_x)
    w = np.zeros(len(train_x[0]) + 1)

    z = []
    if best == 1:
        best_acc = 0
        best_z = []
#         best_epo = 0
        
    if asc:
        ASC = []
        err = 0
        
    
    # reserve develop dataset
    order = list(range(N))
    random.shuffle(order)
    dev_index = order[:int(N * dev_size)]
    dev_index.sort(reverse=True)
    
    [dev_x, dev_y] = develop_data(train_x, train_y, dev_index)
        
    # STD
    for k in range(epochs):
        if GD == False:
            i = random.randint(0, len(train_x)-1)
            g = gradient(train_x[i], train_y[i], w, losstype)
            w += stepSize * g
            if asc:
                a = predict(train_x[i], w)
                if a * train_y[i] <= 0:
                    err += 4
        
        if GD == True:
            gg = np.zeros(len(train_x[0]) + 1)
            for i in range(len(train_x)):
                gg += gradient(train_x[i], train_y[i], w, losstype)
#             gg /= N
            w += stepSize * gg
        
        if (k + 1) % sample == 0:
            if asc:
                ASC.append(err / (k + 1))
            else:
                z.append(np.array(w))
            
            if best == 1 and (epochs - k) / sample <= 100:
                acc = test_accuracy([dev_x, dev_y], w)
                if acc > best_acc:
                    best_acc = acc
                    best_z = np.array(w)
                    best_epo = k + 1
                
    if best == 1:
        test = get_data(1)
        acc = test_accuracy(test, best_z)
        print('best accuracy for test data:',acc,
              '\nbest accuracy for dev data:', best_acc,
             '\nweights:',best_z,
             '\nepochs:',best_epo)
    
    if asc:
        return ASC
    return z
        

def ploting(Y, filename, ylabel, title):
    '''
    plot a figure 
    '''
    X = list(range(1,len(Y)+1))
    X = SAMPLE * np.array(X)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(X, Y, linewidth = '1',color='b')
    ax.set_xlabel('epochs')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(filename+'.png')
    return


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


def plot_ASE(step, epo, loss, gd = False):
    train = get_data(data = 0)
    if train == None:
        return
    ase = stochastic_gradien_descent(train, stepSize = step, 
                                   losstype = loss,
                                   epochs = epo, GD = gd,
                                    asc = True)
    
    losstype = ''
    if loss == 0:
        losstype = 'linear regression'
    elif loss == 1:
        losstype = 'logistic regression'
    ploting(ase,losstype + 'ase'+str(step),
            'average squared error',losstype + 
           ' average squared error with step size '+str(step))
    
    return

def plot_L2norm(step, epo, loss, gd = False):
    train = get_data(data = 0)
    if train == None:
        return
    z = stochastic_gradien_descent(train, stepSize = step, 
                                   losstype = loss,
                                   epochs = epo, GD = gd)
    l2norm = []
    for w in z:
        l2norm.append(np.linalg.norm(w[:-2]))

    losstype = ''
    if loss == 0:
        losstype = 'linear regression'
    elif loss == 1:
        losstype = 'logistic regression'
    ploting(l2norm,losstype + 'l2norm'+str(step),
            'L2 norm', losstype + 
           ' L2 norm with step size '+str(step))
    
    return

def plot_acc(step, epo, loss, gd = False):
    train = get_data(data = 0)
    test = get_data(data = 1)
    if train == None:
        return
    z = stochastic_gradien_descent(train, stepSize = step, 
                                   losstype = loss,
                                   epochs = epo, GD = gd)
    Acc = []
    for w in z:
        Acc.append(test_accuracy(test, w))
    
    losstype = ''
    if loss == 0:
        losstype = 'linear regression'
    elif loss == 1:
        losstype = 'logistic regression'
        
    print ('\nacc:',max(Acc),'\nepo:',
           Acc.index(max(Acc))*SAMPLE,
          '\n weights:', z[Acc.index(max(Acc))])
    ploting(Acc,losstype + 'Acc'+str(step),
            'accuracy', losstype + 
           ' accuracy with step size '+str(step))
    
    return


def best_model(step, epo, loss, gd = False):
    train = get_data(data = 0)
    if train == None:
        return
    
    losstype = ''
    if loss == 0:
        losstype = 'linear regression'
    elif loss == 1:
        losstype = 'logistic regression'
    print('deriving best ' + losstype +'model')
    z = stochastic_gradien_descent(train, stepSize = step, 
                                   losstype = loss,
                                   epochs = epo, best = 1,
                                  GD = gd)
    return


def GD_performance(step, epo, loss):
    train = get_data(data = 0)
    test = get_data(data = 1)
    if train == None:
        return
    
    losstype = ''
    if loss == 0:
        losstype = 'linear regression'
    elif loss == 1:
        losstype = 'logistic regression'
    print('deriving best ' + losstype +'model')
    z = stochastic_gradien_descent(train, stepSize = step, 
                                   losstype = loss,
                                   epochs = epo, best = 1,
                                   GD = True, sample = 10)
    
    Acc = []
    for w in z:
        Acc.append(test_accuracy(test, w))
    
    losstype = ''
    if loss == 0:
        losstype = 'linear regression'
    elif loss == 1:
        losstype = 'logistic regression'

    ploting(Acc,'gd'+str(step),
            'accuracy', 'gd ' + losstype + 
           ' accuracy with step size '+str(step))
    return

if __name__ == "__main__":
    question = input('please enter the question number(2/3): ')
    step = [0.8, 0.001, 0.00001]
    if question == '2':
        problem = input('please enter the question number b.i/b.ii/b.iii/c:')
        if problem == 'b.i':
            for i in step:
                if i == 0.8:
                    plot_ASE(i, 800, 0)
                else:
                    plot_ASE(i, 100000, 0)
            for i in step:
                plot_ASE(i, 100000, 1)
        elif problem == 'b.ii':
            for i in step:
                if i == 0.8:
                    plot_L2norm(i, 500, 0)
                else:
                    plot_L2norm(i, 100000, 0)
            for i in step:
                plot_L2norm(i, 100000, 1) 
        elif problem == 'b.iii':
            print('for linear regression')
            for i in step:
                print('\nfor step size:', i)
                if i == 0.8:
                    plot_acc(i, 800, 0)
                else:
                    plot_acc(i, 100000, 0)
            print('for logistic regression')
            for i in step:
                print('\nfor step size:', i)
                plot_acc(i, 100000, 1)
        elif problem == 'c':
            for i in step:
                print('\nfor step size:', i)
                if i == 0.8:
                    best_model(i, 800, 0)
                else:
                    best_model(i, 100000, 0)
            for i in step:
                print('\nfor step size:', i)
                best_model(i, 100000, 1)
        else:
            print('error problem number!')
    elif question == '3':
        SAMPLE = 10
        gdstep = [0.1, 0.01, 0.001]
        print('GD performance')
        for i in gdstep:
            print('\nfor step size:', i)
            GD_performance(i, 500, 1)

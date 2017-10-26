'''
source code for CSE446 A2 question 2-4
author: Dizhi Ma (dizhim@uw.edu)
'''

'''
part one: perceptron

include the function for training the perceptron.
'''

import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def margin(train_x,w,b):
    '''
    compute the margin
    '''
    wMode=0
    for i in w:
        wMode+=i*i
    wMode=math.sqrt(wMode)
    minDis=100000
    for x in train_x:
        temp=b
        for j in range(len(x)):
            temp+=x[j]*w[j]
        temp=abs(temp)
        if temp/wMode<minDis:
            minDis=temp/wMode
    return minDis
            

def corp_train(train_x,train_y,carve):
    '''
    given a train data and a carve ratio
    return the croped version of the origin data
    '''
    length=len(train_x)
    order=list(range(length))
    random.shuffle(order)
    temp_x=[]
    temp_y=[]
    
    order=order[:int(carve*length)]
    for i in order:
        temp_x.append(train_x[i])
        temp_y.append(train_y[i])
    return [temp_x,temp_y]

def averaged_weight(w,u,c):
    '''
    compute the average weight aw with the original
    weight w, cached weight u, and counter c
    aw=w-u/c
    '''
    aw=[]
    for i in range(len(w)):
        aw.append(w[i]-u[i]/c)
    return aw

def noise_feature_index(weights):
    '''
    given the origin weight return the noise 
    features' index
    '''
    index=[]
    for i in range(len(weights)):
        if abs(weights[i])<0.01:
            index.append(i+1)
    return index
        
def unify_vector(vector):
    '''
    return a given vector's unified version
    '''
    unit=[0]*len(vector)
    mode=0
    for i in vector:
        mode+=i*i
    mode=math.sqrt(mode)
    for i in range(len(vector)):
        unit[i]=vector[i]/mode
    return unit

def abscaling(vector):
    '''
    return a given vector's abs scaling version
    '''
    temp=vector[:]
    for i in range(len(temp)):
        temp[i]=abs(temp[i])
    maxv=max(temp)
    for i in range(len(temp)):
        temp[i]=(temp[i])/(maxv)
    return temp

def all_same(array):
    '''
    given a list determine if all the element are same
    '''
    allSame=False
    for i in array:
        if array.count(i)==len(array):
            allSame=True
    return allSame
    
def mistak_stable(trainError,sameNumber):
    '''
    check if the error rate is stablize
    '''
    length=len(trainError)
    sameError=0
    trainMistakes=trainError[-sameNumber:]
    if length<=sameNumber:
        return False
    else:
        if all_same(trainMistakes):
            return True
        else:
            return False

def error_rate(data,label,w,b):
    '''
    calculate the error rate for certain [w,b]
    '''
    mistakes=0
    for i in range(len(data)):
        a=predict(data[i],w,b)
        if a*label[i]<=0:
            mistakes+=1
    return mistakes/len(data)*100
        
def predict(x,w,b):
    '''
    compute the activation of the example
    '''
    a=0
    for i in range(len(x)):
        a+=w[i]*x[i]
    return a+b

def cached_update(x,y,u,bata,c):
    '''
    update the cached weight u and cached bias
    '''
    bata=bata+c*y
    for i in range(len(x)):
        u[i]+=y*c*x[i]
    return [u,bata]

def update(x,y,w,b):
    '''
    updata weight w, bias b
    '''
    b=b+y
    for i in range(len(x)):
        w[i]+=y*x[i]
    return[w,b]

def PerceptronTrain(train_x,train_y,test_x,test_y,epochs):
    '''
    training the perceptron with the train data and compute 
    the error rate with train data and test data in the end
    of each epochs. the function will return two series of
    error for train data and test data.
    also, the following information will showed:
        error rate achieve(train/test)
        noise feature
        margin
    '''
    sameNumber=100
    trainError=[]
    testError=[]
    print('>>>start trainning Perceptron>>>\n')
    #initialize
    w=[0]*(len(train_x[0]))
    b=0
    for e in range(epochs+1):
        order=list(range(len(train_x)))
        random.shuffle(order)
        for i in order:#range(len(train_x)):
            a=predict(train_x[i],w,b)
            if a*train_y[i]<=0:
                [w,b]=update(train_x[i],train_y[i],w,b)
        trainError.append(error_rate(train_x,train_y,w,b))
        testError.append(error_rate(test_x,test_y,w,b))
        if trainError[-1]==0:
            break
    print('stop at epoch',e)
    scaw=abscaling(w)
    noise=noise_feature_index(scaw)
    print('error rate achieve(train/test):',trainError[-1],
          '/',testError[-1],
          '\nnoise features:',noise)
    if trainError[-1]==0:
        print('margin is:',margin(train_x,w,b))
    print('end training>>>\n')
    return [trainError,testError]

def averaged_perceptron_train(train_x,train_y,test_x,test_y,epochs):
    '''
    training the averaged perceptron with the train data and
    compute the error rate with train data and test data in 
    the end of each epochs. the function will return two 
    series of error for train data and test data
    also, the following information will showed:
        error rate achieve(train/test)
        noise feature
        margin
    '''
    sameNumber=100
    trainError=[]
    testError=[]
    print('>>>strat training Voted Perceptron>>>')
    #initialize
    w=[0]*(len(train_x[0]))
    b=0    
    u=[0]*(len(train_x[0]))
    bata=0
    c=1
    for e in range(epochs+1):
        order=list(range(len(train_x)))
        random.shuffle(order)
        for i in order:
            a=predict(train_x[i],w,b)
            if a*train_y[i]<=0:
                [w,b]=update(train_x[i],train_y[i],w,b)
                [u,bata]=cached_update(train_x[i],train_y[i],u,bata,c)
            c+=1
        meanW=averaged_weight(w,u,c)
        meanB=b-bata/c
        trainError.append(error_rate(train_x,train_y,meanW,meanB))
        testError.append(error_rate(test_x,test_y,meanW,meanB))
        if ((e>sameNumber and all_same(trainError[-sameNumber:]))
            or trainError[-1]==0):
#         if trainError[-1]==0:
            break
    print('stop at epoch',e)
    scaw=abscaling(w)
    noise=noise_feature_index(scaw)
    print('error rate achieve(train/test):',trainError[-1],
          '/',testError[-1],
          '\nnoise features:',noise)
    if trainError[-1]==0:
        print('margin is:',margin(train_x,w,b))
    print('end training>>>\n')
    return [trainError,testError]

def DevelopPerceptronTrain(train_x,train_y,test_x,test_y,epochs,carve):
    '''
    training the averaged perceptron with the develop data
    which is a corped version of the train data, the corp
    ratio is decided by the parameter 'carve'. In the end 
    of each epoch, the error rate for develop data, train 
    data, and test data will be calculated. The function
    will return three series of error for develop, train,
    and test data.
    also, the following information will showed:
        error rate achieve(develop/train/test)
        noise feature
        margin
    '''
    develop=[]
    develop_y=[]
    order=list(range(len(train_x)))
    random.shuffle(order)
    devIndex=order[:int(len(train_x)*carve)]
    for i in devIndex:
        develop.append(train_x[i])
        develop_y.append(train_y[i])
    devIndex.sort(reverse=True)
    for i in devIndex:
        del train_x[i]
        del train_y[i]
    developError=[]
    trainError=[]
    testError=[]
    print('>>>start trainning Perceptron with develop data>>>\n')
    #initialize
    w=[0]*(len(train_x[0]))
    b=0
    for e in range(epochs+1):
        order=list(range(len(train_x)))
        random.shuffle(order)
        for i in order:
            a=predict(train_x[i],w,b)
            if a*train_y[i]<=0:
                [w,b]=update(train_x[i],train_y[i],w,b)
        developError.append(error_rate(develop,develop_y,w,b))
        trainError.append(error_rate(train_x,train_y,w,b))
        testError.append(error_rate(test_x,test_y,w,b))
        if e>0 and e%100==0:
            print('for epoch',e,
                  'error rate achieve(develop/train/test):',developError[-1],
                  '/',trainError[-1],'/',testError[-1])
        if developError[-1]==0 or trainError[-1]==0:
            break
    print('stop at epoch',e)
    scaw=abscaling(w)
    noise=noise_feature_index(scaw)
    print('error rate achieve(develop/train/test):',developError[-1],
          '/',trainError[-1],'/',testError[-1],
          '\nnoise features:',noise,)
    if trainError[-1]==0:
        print('margin is:',margin(train_x,w,b))
    print('end training>>>\n')
    return [developError[-1],trainError[-1],testError[-1]]

'''
part two: PCA algorithm
'''

def zeroMean(dataMat):        
    meanVal=np.mean(dataMat,axis=0)
    newData=dataMat-meanVal  
    return newData,meanVal  

def percentage2n(eigVals,percentage):  
    sortArray=np.sort(eigVals)   
    sortArray=sortArray[-1::-1] 
    arraySum=sum(sortArray)  
    tmpSum=0  
    num=0  
    for i in sortArray:  
        tmpSum+=i  
        num+=1  
        if tmpSum>=arraySum*percentage:  
            return num  
        
def pca(dataMat,testMat,percentage):  
    newData,meanVal=zeroMean(dataMat)  
    covMat=np.cov(newData,rowvar=0)      
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    fig1 = plt.figure()
    ax = fig1.add_subplot(1,1,1)
    X=np.arange(len(eigVals))+1
    ax.bar(X,eigVals/sum(eigVals),width = 0.35)
    ax.set_title('eigenvalues/sum(eigenvalues)')
    fig1.savefig('eigenvalues.png')
    n=percentage2n(eigVals,percentage/100)
    print('with',percentage,'% of the eigen value,', 
          'lower the feature to ',
          n,'-dimension')              
    eigValIndice=np.argsort(eigVals)              
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]     
    n_eigVect=eigVects[:,n_eigValIndice]         
    lowDDataMat=newData*n_eigVect        
    newTest,meanTest=zeroMean(testMat)
    lowTest=newTest*n_eigVect
    lowDDataMat=np.array(lowDDataMat).tolist()
    lowTest=np.array(lowTest).tolist()
    return lowDDataMat,lowTest 

'''
part three: automation initiate the perceptron
'''

def read_A2(filepath):
    '''
    function for reading the A2 dataset 2-9. given 
    a filepath, data and labels will be return. 
    '''
    df=pd.read_csv(filepath,header=None,sep='\t')
    temp=np.array(df)
    data=[]
    labels=[]
    for example in temp:
        data.append(example[1:].tolist())
        labels.append(example[0])
    return [data,labels]

def corp_noise678(data):
    '''
    corp out the noise for each example in the data6-8
    and return the corped version of the data
    '''
    temp=[]
    noise=[18,17,11,9]
    for i in data:
        for j in noise:
            del i[j]
        temp.append(i)
    return temp

def plot_mistake(trainError,testError,filename):
    '''
    plot a figure for error rate vs epochs
    '''
    length=len(trainError)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(length),trainError,
            linewidth = '1',color='b',label='training data error')
    ax.plot(range(length),testError,
            linewidth = '1',color='r',label='test data error')
    ax.set_xlabel('epochs')
    ax.set_ylabel('error rate(%)')
    ax.set_title('error rate for '+filename)
#     ax.set_ylim(0,10)
    ax.legend()
    fig.savefig(filename+'.png')

def plot678(epochs=3000,ptype=0,carve=1,develop=False):
    '''
    return a training perceptron performance plot on the 
    test 6 data with the improved dataset.
    
    epochs- maximum training epoch for the perceptron,defualt
            number is 3000
    ptype - perceptron type. 0 is the normal perceptron, 1 is
            for averaged perceptron
    carve - carve out ratio for the data.
    develop - form a develop versoin for the data. With the 
            develop data, the function will only return the 
            error rate but not a plot
    '''
    train_x=[]
    train_y=[]
    for i in [6,7,8]:
        trainpath='A2.'+str(i)+'.train.tsv'
        [trainx,trainy]=read_A2(trainpath)
        train_x+=trainx
        train_y+=trainy
    if carve!=1:
        [train_x,train_y]=corp_train(train_x,train_y,carve)
#         print('crop!')

    testpath='A2.6.test.tsv'
    [test_x,test_y]=read_A2(testpath)

    train_x=corp_noise678(train_x)
    test_x=corp_noise678(test_x)

    if develop:
        result=DevelopPerceptronTrain(train_x,train_y,
                                      test_x,test_y,
                                      epochs,carve)        
    elif ptype==0:
            [trainError,testError]=PerceptronTrain(train_x,train_y,
                                                   test_x,test_y,epochs)
            plot_mistake(trainError,testError,'combine A2.6,7,8')
    elif ptype==1:
        [trainError,testError]=averaged_perceptron_train(train_x,train_y,
                                                         test_x,test_y,epochs)
        plot_mistake(trainError,testError,'combine A2.6,7,8'+'voted')
    else:
        print('ptype must be 1 or 0')
        return 
            
    return
        

def A2plot(i,epochs=3000,ptype=0,carve=1,PCA=0):
    '''
    return one or more plots and final training result
    of perceptron performance for A2 dataset.
    
    i     - the dataset number
    epoch - the maxmum training epoch. default value is
            3000
    ptype - perceptron type. 0 is the normal perceptron, 1 is
            for averaged perceptron
    carve - carve out ratio for the data, default value is 1.
            when it is not 1, a develop data will be created
            and used for training another perceptron. in 
            this way one the final result will be print,
            no plot will be generate
    PCA   - lower the dimension of the train data. default value
            is 0, means not to PCA. when using PCA, set it
            as remain percetage eigenvalue you want. 
            eg. 0.9
    '''
    
    print('\ndataset:',i)
    trainpath='A2.'+str(i)+'.train.tsv'
    testpath='A2.'+str(i)+'.test.tsv'
    [train_x,train_y]=read_A2(trainpath)
    [test_x,test_y]=read_A2(testpath)
    if PCA!=0:
        print('pca!')
        train_x,test_x=pca(train_x,test_x,PCA)


    if carve!=1:
        result=DevelopPerceptronTrain(train_x,train_y,
                                      test_x,test_y,
                                      epochs,carve)

    elif ptype==0:
        [trainError,testError]=PerceptronTrain(train_x,train_y,
                                               test_x,test_y,epochs)
        plot_mistake(trainError,testError,'A2.'+str(i))
    elif ptype==1:
        [trainError,testError]=averaged_perceptron_train(train_x,train_y,
                                                         test_x,test_y,epochs)
        plot_mistake(trainError,testError,'A2.'+str(i)+'voted')
    else:
        print('ptype must be 1 or 0')
        return
        
    return

'''
part four: main function
'''

if __name__=="__main__":
    while(True):
        question=input('please enter the question number(2.1/2.5/surprise/3/4): ')
        if question=='2.1':
            i=int(input('enter the dataset number: '))
            A2plot(i)
            print('plot A2.'+str(i)+'.png has been created>>>\n')
        elif question == '2.5':
            i=int(input('enter the dataset number: '))
            A2plot(i,epochs=1000,carve=0.2)
        elif question == '3':
            i=int(input('enter the dataset number: '))
            A2plot(i,ptype=1)
            print('plot A2.'+str(i)+'voted.png has been created>>>\n')
        elif question == 'surprise' or question == '678':
            plot678()
            print('A2.combine A2.6,7,8.png has been created>>>\n')
        elif question == '4':
            i=int(input('enter the dataset number: '))
            if input('type y for averaged perceptron:')=='y':
                A2plot(i,PCA=95,ptype=1)
                print('plot A2.'+str(i)+'voted.png has been created>>>\n')
            else: 
                A2plot(i,PCA=95)
                print('plot A2.'+str(i)+'.png has been created>>>\n')
        else:
            print('wrong question number! must be 2.1 or 2.5 or surprise or 3 or 4')
        if input('type y for exit:') == 'y':
            break
            
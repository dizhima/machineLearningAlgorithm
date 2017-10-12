'''
DECISION TREE TRAIN ALGORITHM:
    DecisionTreeTrain(data,remain features):
        guess <-- most frequent answer(label) in data //default answer for the data
        if (the labels in data are unambiguous):
            return LEAF(guess)
        else if (remain features is empty):
            return LEAF(guess)
        else:
            for all f belong to remain features:
                NO <-- the subset of data on which f=no
                YES <-- the subset of data on which f=yes
                score[f] <-- # of majority vote answers in NO
                            +# of majority vote answers in YES
                            //accuracy we would get if only queried on f
            end for.
            f <-- the feature with maximal score(f)
            NO <-- the subset of data on which f=no
            YES <-- the subset of data on which f=yes
            remain features = remain features - f
            left <-- DecisionTreeTrain(NO, remain features)
            right <-- DecisionTreeTrain(YES, remain features)
            return NODE(f,left,right)
        end if.
        
DECISION TREE TEST ALGORITHM:
    DecisionTreeTrain(tree,test point):
        if (tree is of the form LEAF(guess)):
            return guess
        else if (tree is of from NODE(f,left,right)):
            if (f=no in the test point):
                return DecisionTreeTrain(left,test point)
            else
                return DecisionTreeTrain(right,test point)
            end if
        end if

references:
Hal Daume. A Course in Machine Learning (v0.9)
'''

'''
PART ONE:
TRANSFORM THE ORIGIN DATA TO PREFER FORMAT

THE PREFER DATA FORMAT CAN BE DESCRIBED AS A Mx(N+1) MATRIX WHICH CONTAINS M
SAMPLES, EACH SAMPLE HAS N FEATURES
EACH ROW IS A SAMPLE, THE FIRST N TERMS ARE THE VALUE CORRESPOND TO THE FEATURE
AND LAST TERM IS THE LABEL

'''

dataSet=[[24,40,1],
        [53,52,0],
        [23,25,0],
        [25,77,1],
        [32,48,1],
        [52,110,1],
        [22,38,1],
        [43,44,0],
        [52,27,0],
        [48,65,1]]
a1=[24,53,23,25,32,52,22,43,52,48]
a2=[40,52,25,77,48,110,38,44,27,65]

a1.sort()
temp=[]
for i in range(len(a1)-1):
    if a1[i]!=a1[i+1]:
        temp.append((a1[i]+a1[i+1])/2)
a1=temp

a2.sort()
temp=[]
for i in range(len(a2)-1):
    if a2[i]!=a2[i+1]:
        temp.append((a2[i]+a2[i+1])/2)

a2=temp
feat=[a1,a2]
data=[]
for example in dataSet:
    featValue=[]
    for i in range(len(feat)):
        for j in range(len(feat[i])):
            if example[i]>feat[i][j]:
                featValue.append(1)
            else:
                featValue.append(0)
    data.append(featValue)

for i in range(len(data)):
    data[i].append(dataSet[i][-1])

feature=[]
for i in range(2):
    for example in feat[i]:
        if i==0:
            feature.append('age>'+str(example))
        else:
            feature.append('salary>'+str(example))    
            
'''
PART TWO:
DECISION TREE TRAIN ALGORITHM
'''
import numpy as np
import math

def split_data(data,i):
    no=[]
    yes=[]
    for dataSet in data:
        if dataSet[i]==0:# go no
            no.append(dataSet[:i]+dataSet[i+1:])
        if dataSet[i]==1:#go yes
            yes.append(dataSet[:i]+dataSet[i+1:])
    return [no,yes]

def score(splited):
    no=splited[0]
    yes=splited[1]
    n0=p0=n1=p1=0
    for data in no:
        if data[-1]==0:
            n0+=1
        else:
            p0+=1
    for data in yes:
        if data[-1]==0:
            n1+=1
        else:
            p1+=1
    #with accuracy
    gpa=(max(n0,p0)+max(n1,p1))/(n0+p1+n1+p0) if (len(no)!=0 and len(yes)!=0) else 0  
    # with mutual information
#     D=n0+n1+p0+p1
#     n=n0+n1
#     p=p0+p1
#     t1=n0/D*math.log(D*n0/((n0+p0)*n)) if n0>0 else 0
#     t2=p0/D*math.log(D*p0/((n0+p0)*p)) if p0>0 else 0
#     t3=n1/D*math.log(D*n1/((n1+p1)*n)) if n1>0 else 0
#     t4=p1/D*math.log(D*p1/((n1+p1)*p)) if p1>0 else 0
#     gpa=(t1+t2+t3+t3)
    return gpa

def build_DTree_Greedily(data, features):
    classes=['no','yes']
    labelList=[label[-1] for label in data]
    counts = np.bincount(labelList)
    guess=np.argmax(counts)#most frequent label in data
    slabel=True#index show if the labels are same, False means not all the same
    for i in range(len(data)-1):
        if data[i][-1] != data[i+1][-1]:
            slabel=False#index show if the labels are same, False means not all the same
            break
    if(slabel):#not sames labels and data not empty
        return classes[guess]
    elif len(data[0])==1:#no more features
        return classes[guess]
    else:
        scores=[]
        for i in range(len(features)):
            noYes=split_data(data,i)
            scores.append(score(noYes))
        f=scores.index(max(scores))#feature with maximal scorea,f
        no=split_data(data,f)[0]#subset of N0
        yes=split_data(data,f)[1]#subset of YES
        newLabel=features[f]  
        Tree={newLabel:{}}
        features=features[:f]+features[f+1:]
        Tree[newLabel][0]=build_DTree_Greedily(no,features)#left DTreeTrain
        Tree[newLabel][1]=build_DTree_Greedily(yes,features)#right DTreeTrain
        return Tree

'''
PART THREE:
BUILD THE DECISION TREE
'''
build_DTree_Greedily(data,feature)

'''
PART FOUR:
DECISION TREE TEST ALGORITHM
'''
def classify(tree,label,testVec):
    firstFeat=list(tree.keys())[0]
    secondDict=tree[firstFeat]
    labelIndex=label.index(firstFeat)
    for key in secondDict.keys():
        if testVec[labelIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],label,testVec)
            else:
                classLabel=secondDict[key]
return classLabel

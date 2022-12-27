#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
#from sklearn.utils import shuffle
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM,GRU
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

dataset =pd.read_csv('dataset.csv')
#dataset = shuffle(dataset)
X=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13].values

def Multilabelencoder(X,k):
    X[:,k]= LabelEncoder().fit_transform(X[:,k])
    return X
for i in range(1,4):
    X=Multilabelencoder(X,i)
y= LabelEncoder().fit_transform(y)


X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=0)

sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)






s=time.time()

import xgboost as xgb
classifier1=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
classifier1.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
classifier2.fit(X_train,y_train)

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
classifier3 = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
classifier3.fit(X_train, y_train)

from sklearn.ensemble import AdaBoostClassifier
classifier4 = AdaBoostClassifier(random_state=1)
classifier4.fit(X_train, y_train)

from sklearn.ensemble import GradientBoostingClassifier
classifier5= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
classifier5.fit(X_train, y_train)

from catboost import CatBoostClassifier
classifier6=CatBoostClassifier()
categorical_features_indices = np.where(dataset.dtypes != np.float)[0]
classifier6.fit(X_train,y_train,eval_set=(X_test, y_test))


import lightgbm as lgb
train_data=lgb.Dataset(X_train,label=y_train)
params = {'learning_rate':0.001}
classifier7 = lgb.train(params, train_data, 100) 

from sklearn.ensemble import VotingClassifier
classifier8 = VotingClassifier(estimators=[('classifier1', classifier1), ('classifier2', classifier2),('classifier3', classifier3),('classifier4', classifier4),('classifier5', classifier5),('classifier6', classifier6)], voting='hard')
classifier8.fit(X_train,y_train)

e=[classifier1,classifier2,classifier3,classifier4,classifier5,classifier6,classifier7,classifier8]







X_first=X_test
from sklearn.decomposition import PCA
pca= PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance= pca.explained_variance_ratio_
    
import xgboost as xgb
classifier1=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
classifier1.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
classifier2.fit(X_train,y_train)

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
classifier3 = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
classifier3.fit(X_train, y_train)

from sklearn.ensemble import AdaBoostClassifier
classifier4 = AdaBoostClassifier(random_state=1)
classifier4.fit(X_train, y_train)

from sklearn.ensemble import GradientBoostingClassifier
classifier5= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
classifier5.fit(X_train, y_train)

from catboost import CatBoostClassifier
classifier6=CatBoostClassifier()
categorical_features_indices = np.where(dataset.dtypes != np.float)[0]
classifier6.fit(X_train,y_train,eval_set=(X_test, y_test))

import lightgbm as lgb
train_data=lgb.Dataset(X_train,label=y_train)
params = {'learning_rate':0.001}
classifier7 = lgb.train(params, train_data, 100) 

from sklearn.ensemble import VotingClassifier
classifier8 = VotingClassifier(estimators=[('classifier1', classifier1), ('classifier2', classifier2),('classifier3', classifier3),('classifier4', classifier4),('classifier5', classifier5),('classifier6', classifier6)], voting='hard')
classifier8.fit(X_train,y_train)

f=[classifier1,classifier2,classifier3,classifier4,classifier5,classifier6,classifier7,classifier8]
g=["XG Boost","Random Forest","Bagging Classifier - Decision Tree","Ada Boost","Gradient Boost","Cat Boost","Light GBM","Voting Classifier"]





def abc(h):
    warnings.filterwarnings("ignore")
    classifier=e[h]
    Y_pred = classifier.predict(X_first)
    #
    for i in range(0,len(Y_pred)):
        if Y_pred[i]>=0.5:
            Y_pred[i]=1
        else: 
            Y_pred[i]=0
    from  sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,Y_pred)
    #
    print("Accuracy of the "+g[h]+" Model is : ",(cm[0][0]+cm[1][1])*100/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]))
    print("Precision of the "+g[h]+" Model is : ",(cm[0][0])*100/(cm[0][0]+cm[1][0]))
    print("Recall of the "+g[h]+" Model is : ",(cm[0][0])*100/(cm[0][0]+cm[0][1]))
    #

    #
    classifier=f[h]

    
    pred_label = classifier.predict(X_test)
    #pred_label=Y_pred
    #print(pred_label[:50])
    '''
    list1=[]
    for i in range(len(pred_label)):
        if pred_label[i]== 0:
            list1.append('Normal')
        elif pred_label[i]==1:
            list1.append('Anomaly')

    data=collections.Counter(list1)
    ta = list(data.keys())
    cnt = list(data.values())
    fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
    plt.bar(ta, cnt, color ='maroon',width = 0.4)
    
    plt.xlabel("Predicted attack types")
    plt.ylabel("Count")
    plt.title("Analysis of {}".format(e[h]))
    plt.show()
    '''

    

    
def ANN():
    ANN_dataset=[[0]*8 for i in range(5039)]
    for j in range(8):
        classifier=e[j]
        Y_predn = classifier.predict(X_first)
        for i in range(0,len(Y_predn)):
            if Y_predn[i]>=0.5:
                ANN_dataset[i][j]=1
            else: 
                ANN_dataset[i][j]=0    
    A=pd.DataFrame(ANN_dataset)
    b=y_test
    #print(A.shape,b.shape)
    print(A)

    X_trainn, X_testn, y_trainn, y_testn =train_test_split(A,b,test_size=0.2,random_state=0)
    

    #kernel_initializerializing ANN
    classifier = Sequential()
    
    #Adding input layer and first hidden layer
    #classifier.add(GRU(1,activation="tanh"))
    #classifier.add(Dense(1,kernel_initializer='he_uniform',activation='sigmoid',input_dim=8))
    classifier.add(Dense(1,kernel_initializer='uniform',activation='relu',input_dim=8))
    classifier.add(Dropout(0.1))
    
    
    #classifier.add(Dense(output_dim=1,kernel_initializer='uniform',activation='sigmoid'))
    
    #Compiling ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    
    #Fitting ANN to training set
    classifier.fit(X_trainn,y_trainn,batch_size=10)
    
    #Making Prediction and Evaluating model
    #prediction
    y_pred = classifier.predict(X_testn)
    y_pred = (y_pred > 0.5)
    
    #confusion matrix
    
    cm=confusion_matrix(y_testn,y_pred)
    
    #Evaluating ANN

    def build_classifier():
        classifier = Sequential()
        
        classifier.add(Dense(1,kernel_initializer='uniform',activation='relu',input_dim=8))
        #classifier.add(LSTM(1,activation="tanh",recurrent_activation="hard_sigmoid"))
        #classifier.add(Dense(output_dim=1,kernel_initializer='uniform',activation='sigmoid'))
        classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
        return classifier
    
    classifier = KerasClassifier(build_fn = build_classifier,batch_size=10,nb_epoch=1)
    accuracies = cross_val_score(estimator=classifier,X=X_trainn, y=y_trainn,cv=10)
    mean=accuracies.mean()
    variance=accuracies.std()
    print(mean,variance)
    #Tuning the ANN

    
    def build_classifier(optimizer):
        classifier = Sequential()
        classifier.add(Dense(4,kernel_initializer='uniform',activation='relu',input_dim=8))
        classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
        classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
        return classifier
    
    classifier = KerasClassifier(build_fn = build_classifier)

    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # param_grid = dict(optimizer=optimizer)
    grid_param={'batch_size':[25,32],
                'nb_epoch':[100,500],
                'optimizer':['adam','rmsprop']
                }

    grid_search = GridSearchCV(estimator=classifier,
                        #param_grid=param_grid,
                         param_grid=grid_param,
                         scoring='accuracy',
                         cv=5)

    grid_search = grid_search.fit(X_trainn,y_trainn)
    best_parameters=grid_search.best_params_
    print(grid_search.best_score_)
    
import multiprocessing as mp
def master():
    if __name__=='__main__':
        while(1):
            warnings.filterwarnings("ignore")
            s=int(input("""\nWelcome to Network Intrusion Detection System
                    Make your choice :
                    0. XG Boost
                    1. Random Forest
                    2. Bagging Classifier - Decision Tree
                    3. Ada Boost
                    4. Gradient Boost
                    5. Cat Boost
                    6. Light GBM
                    7. Voting Classifier
                    8. Run all Algorithms
                    9. Quit\n"""))
            if(s<8):
                abc(s)

            elif(s==8):
                s=int(input("""Select how you want to run it :
                                0. Serially
                                1. Artificial Neural Network
                                2. Go Back\n"""))
                if(s==0):
                    for i in range(8):
                        abc(i)
                elif(s==1):
                    ANN()
                elif(s==2):
                    pass
                
                else:
                    print("Invalid Choice :( Try Again")
            elif(s==9):
                break;
            else:
                print("Invalid Choice :( Try Again")
    else:
        pass

def parallel():
    p = mp.Pool(8)
    for h in range(4):
        p.apply_async(abc, args=(h,))
    p.close()
    p.join()

master()






























    
    
from sklearn import tree
from sklearn import grid_search

import pandas
import numpy
import math

dataframe_train = pandas.read_csv("training.csv", header=None)
dataset_train = dataframe_train.values

dataframe_test = pandas.read_csv("test.csv", header=None)
dataset_test = dataframe_test.values

for i in range(250001):
    if dataset_train [i,32]=='s':
        dataset_train [i,32]='1'
    else:
        dataset_train [i,32]='0'
        
print dataset_train.shape
print dataset_test.shape
X_test = dataset_train[1:10001,:31].astype(float)
Y_test = dataset_train[1:10001,32].astype(float)

X_train = dataset_train[10001:200001,:31].astype(float)
Y_train = dataset_train[10001:200001,32].astype(float)

W = dataset_train[1:10001,31].astype(float)

def AMS(estimator,X,y):
    G=estimator.predict(X)

    cg=0.
    sg=0
    bg=0
    for i in range(len(y)):
        if G[i]== y[i]:
            cg+=1
        if G[i]==1:
            if y[i]==1:
                sg= sg + W[i]
            if y[i]==0:
                bg= bg + W[i]
    score_G=cg*100/len(y) 
    radicandG = 2 *( (sg+bg+10) * math.log (1.0 + sg/(bg+10)) -sg)
    AMS_G = math.sqrt(radicandG)
    print ('Pourcentage = ',score_G,'%')
    print ('AMS =',AMS_G)
    return AMS_G

reg = tree.DecisionTreeRegressor ()
reg = reg.fit(X_train,Y_train)
AMS(reg,X_test,Y_test)
#71.07
#AMS = 0.273223317766

clf = tree.DecisionTreeClassifier ()
clf = clf.fit(X_train,Y_train)
AMS(clf,X_test,Y_test)
#70.65
#AMS = 0.272780740678


from sklearn.ensemble import GradientBoostingClassifier

gradient = GradientBoostingClassifier(n_estimators=50, random_state=0)
gradient = gradient.fit(X_train,Y_train)
AMS(gradient,X_test,Y_test)
#82.33
#AMS = 0.51961354779

#Avec un gridsearch sur le n_estimators et max_depth
W = dataset_train[10001:200001,31].astype(float)
parameters = {'n_estimators':[50,100,250,500], 'max_depth':[2,3,5,8,10]}

gradient = GradientBoostingClassifier()

best_grad = grid_search.GridSearchCV(gradient, parameters, scoring=AMS)
best_grad.fit(X_train, Y_train)
print("Best: %f using %s" % (best_grad.best_score_, best_grad.best_params_))
for params, mean_score, scores in best_grad.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    
gradient = best_grad.best_estimator_

#Résultat : Best: 143.273124 using {'n_estimators': 50, 'max_depth': 3}

#En prenant comme ensemble de test le dataset_test, afin de comparer les résultats à ceux sur Kaggle
X_test = dataset_test[:,:31].astype(float)

print(gradient.predict_proba(X_train)[1,:])
print(Y_train[1])

T=gradient.predict_proba(X_test)[:,1]
print (T)
print (T.shape)
temp = T.argsort()
ranks = numpy.empty(len(T), int)
ranks[temp] = numpy.arange(len(T)) +1

S=[]
for i in range(len(T)):
    if T[i]>0.5:
        S.append('s')
    else:
        S.append('b')
x=1
res = [["EventId","RankOrder","Class"]]
for i in range(len(G)):
    res.append([int(X_test[i,0]),ranks[i],S[i]])
    
print len(res)       
import csv
c= csv.writer(open("result_grad.csv","wb"))

for i in res:
    c.writerow(i)
#Score AMS donné par Kaggle : 2.64820


from sklearn.ensemble import RandomForestClassifier

X_test = dataset_train[1:10001,:31].astype(float)

forest = RandomForestClassifier(n_estimators = 50,random_state=0)
forest = forest.fit(X_train,Y_train)
AMS(forest,X_test,Y_test)
#83.2
#AMS = 0.536340014348

parameters = {'n_estimators':[3, 5, 10, 20, 50, 100], 'max_depth':[2,3,5,8,10]}

forest = RandomForestClassifier()

best_forest = grid_search.GridSearchCV(forest, parameters, scoring=AMS)
best_forest.fit(X_train, Y_train)
print("Best: %f using %s" % (best_forest.best_score_, best_forest.best_params_))
for params, mean_score, scores in best_forest.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    
forest = best_forest.best_estimator_
#Résultat pour la Random Forest : Best: 146.466420 using {'n_estimators': 100, 'max_depth': 10}

X_test = dataset_test[:,:31].astype(float)

print(forest.predict_proba(X_train)[1,:])
print(Y_train[1])

T=forest.predict_proba(X_test)[:,1]
print (T)
print (T.shape)
temp = T.argsort()
ranks = numpy.empty(len(T), int)
ranks[temp] = numpy.arange(len(T)) +1

S=[]
for i in range(len(T)):
    if T[i]>0.5:
        S.append('s')
    else:
        S.append('b')
x=1
res = [["EventId","RankOrder","Class"]]
for i in range(len(G)):
    res.append([int(X_test[i,0]),ranks[i],S[i]])
    
print len(res)       
import csv
c= csv.writer(open("result_forest.csv","wb"))

for i in res:
    c.writerow(i)

#Score Kaggle : 2.82174

from sklearn.ensemble import AdaBoostClassifier

X_test = dataset_train[1:10001,:31].astype(float)

ada = AdaBoostClassifier(base_estimator=reg,
                         n_estimators=50, 
                         learning_rate=0.1, random_state=0)
ada = ada.fit(X_train, Y_train)
AMS(ada,X_test,Y_test)
#82 with 50 estimators, AMS = 0.505636131249

parameters = {'n_estimators':[3, 5, 10, 20, 50, 100]}

ada = AdaBoostClassifier()

best_ada = grid_search.GridSearchCV(ada, parameters, scoring=AMS)
best_ada.fit(X_train, Y_train)
print("Best: %f using %s" % (best_ada.best_score_, best_ada.best_params_))
for params, mean_score, scores in best_ada.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    
ada = best_ada.best_estimator_
#Résultat pour Adaboost Best: 131.665896 using {'n_estimators': 20}

X_test = dataset_test[:,:31].astype(float)

print(ada.predict_proba(X_train)[1,:])
print(Y_train[1])

T=ada.predict_proba(X_test)[:,1]
print (T)
print (T.shape)
temp = T.argsort()
ranks = numpy.empty(len(T), int)
ranks[temp] = numpy.arange(len(T)) +1

S=[]
for i in range(len(T)):
    if T[i]>0.5:
        S.append('s')
    else:
        S.append('b')
x=1
res = [["EventId","RankOrder","Class"]]
for i in range(len(G)):
    res.append([int(X_test[i,0]),ranks[i],S[i]])
    
print len(res)       
import csv
c= csv.writer(open("result_ada.csv","wb"))

for i in res:
    c.writerow(i)

#Score Kaggle : 2.28066

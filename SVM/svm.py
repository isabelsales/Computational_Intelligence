import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.mlab as mlab

# Quaisquer resultados gravados no diretório atual são salvos como saída.
open_file = pd.read_csv("banana3.csv",sep=",")
#print(open_file.head())
#print(open_file.shape)

#print(open_file.isnull().values.any())

#print(open_file.describe())

#Agora, usando a biblioteca Matplotlib, plotando os dois recursos no gráfico de dispersão

import matplotlib.pyplot as plt

x = open_file[["At1"]]
y = open_file[["At2"]]
c = open_file[["Class"]]

for i in c:
    col = np.where(c[i]==-1,'#585858',"#D0A9F5")

plt.scatter(x, y, c=col)
plt.show()

#Agora, plotar os dois recursos na matriz de correlação

file = open_file[['At1','At2']]
#print(file.head())

correlation = file.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=-1, vmax=1)
fig.colorbar(cax)
names=["At1","At2"]
ticks = np.arange(0,2,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

#Agora, usando a biblioteca sklearn, importamos o train_test_test da validação
# cruzada e dividimos o conjunto de dados original em conjunto de dados de treinamento e teste (70,30)

from sklearn.cross_validation import train_test_split
train,test = train_test_split(open_file,test_size=0.3)
features_train = train[['At1','At2']]
features_test = test[['At1','At2']]
labels_train = train.Class
labels_test = test.Class
print(labels_test.head())
print(train.shape)
print(test.shape)


#Agora, podemos usar nossos algoritmos de aprendizado de máquina para brincar com treinamento e conjunto de dados de teste.
# Começamos com o classificador "Naive Bayes - Gaussian"

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
training = clf.fit(features_train,labels_train)
predictions = clf.predict(features_test)
print("Previsão: ",predictions)
print("Precisão do classificador Naive Bayes - Gaussian:",clf.score(features_test,labels_test))

print("-----------------------------------------------------------------------------------------------------\n")

#"Support Vector Machine" com kernel Polinomial
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
clf = SVC(kernel='poly', degree=8)
training = clf.fit(features_train,labels_train)
predictions = clf.predict(features_test)
print("Previsão: ",predictions)
print("Precisão com kernel polinomial:",clf.score(features_test,labels_test))

X = open_file.drop('Class', axis=1)
y = open_file['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=2)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 2: ", total/2)
print("Eout: ", 1 - (total/2))
scores = cross_val_score(clf, X, y, cv=5)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 5: ", total/5)
print("Eout: ", 1 - (total/5))
scores = cross_val_score(clf, X, y, cv=10)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 10: ", total/10)
print("Eout: ", 1 - (total/10))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print("-----------------------------------------------------------------------------------------------------\n")

#"Support Vector Machine" com kernel sigmoid (gamma=1)
from sklearn.svm import SVC
clf = SVC(kernel='sigmoid', gamma=1)
training = clf.fit(features_train,labels_train)
predictions = clf.predict(features_test)
print("Previsão: ",predictions)
print("Precisão com kernel sigmoid (gamma=1):",clf.score(features_test,labels_test))

X = open_file.drop('Class', axis=1)
y = open_file['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=2)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 2: ", total/2)
print("Eout: ", 1 - (total/2))
scores = cross_val_score(clf, X, y, cv=5)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 5: ", total/5)
print("Eout: ", 1 - (total/5))
scores = cross_val_score(clf, X, y, cv=10)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 10: ", total/10)
print("Eout: ", 1 - (total/10))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print("-----------------------------------------------------------------------------------------------------\n")

#"Support Vector Machine" com kernel sigmoid (gamma=0.5)
from sklearn.svm import SVC
clf = SVC(kernel='sigmoid', gamma=0.5)
training = clf.fit(features_train,labels_train)
predictions = clf.predict(features_test)
print("Previsão: ",predictions)
print("Precisão com kernel sigmoid (gamma=0.5):",clf.score(features_test,labels_test))

X = open_file.drop('Class', axis=1)
y = open_file['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=2)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 2: ", total/2)
print("Eout: ", 1 - (total/2))
scores = cross_val_score(clf, X, y, cv=5)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 5: ", total/5)
print("Eout: ", 1 - (total/5))
scores = cross_val_score(clf, X, y, cv=10)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 10: ", total/10)
print("Eout: ", 1 - (total/10))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print("-----------------------------------------------------------------------------------------------------\n")

#"Support Vector Machine" com kernel sigmoid (gamma=0.01)
from sklearn.svm import SVC
clf = SVC(kernel='sigmoid', gamma=0.01)
training = clf.fit(features_train,labels_train)
predictions = clf.predict(features_test)
print("Previsão: ",predictions)
print("Precisão com kernel sigmoid (gamma=0.01):",clf.score(features_test,labels_test))

X = open_file.drop('Class', axis=1)
y = open_file['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=2)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 2: ", total/2)
print("Eout: ", 1 - (total/2))
scores = cross_val_score(clf, X, y, cv=5)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 5: ", total/5)
print("Eout: ", 1 - (total/5))
scores = cross_val_score(clf, X, y, cv=10)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 10: ", total/10)
print("Eout: ", 1 - (total/10))


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print("-----------------------------------------------------------------------------------------------------\n")

#Agora, usando nosso segundo classificador como "Support Vector Machine" com kernel = "rbf"

from sklearn import svm
clf = svm.SVC(kernel='rbf')
training = clf.fit(features_train,labels_train)
predictions = clf.predict(features_test)
print("Previsão: ",predictions)
print("Precisão com kernel rbf:",clf.score(features_test,labels_test))

X = open_file.drop('Class', axis=1)
y = open_file['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=2)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 2: ", total/2)
print("Eout: ", 1 - (total/2))
scores = cross_val_score(clf, X, y, cv=5)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 5: ", total/5)
print("Eout: ", 1 - (total/5))
scores = cross_val_score(clf, X, y, cv=10)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 10: ", total/10)
print("Eout: ", 1 - (total/10))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print("-----------------------------------------------------------------------------------------------------\n")

#"Support Vector Machine" com kernel="linear".

from sklearn import svm
clf = svm.SVC(kernel='linear', C=1)
training = clf.fit(features_train,labels_train)
predictions = clf.predict(features_test)
print("Previsão: ",predictions)
print("Precisão com kernel linear:",clf.score(features_test,labels_test))

X = open_file.drop('Class', axis=1)
y = open_file['Class']

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
y_pred = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=2)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 2: ", total/2)
print("Eout: ", 1 - (total/2))
scores = cross_val_score(clf, X, y, cv=5)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 5: ", total/5)
print("Eout: ", 1 - (total/5))
scores = cross_val_score(clf, X, y, cv=10)
total = 0
for i in scores:
    total  +=i
print("Scores:", scores)
print("Score K-fold cross-validation = 10: ", total/10)
print("Eout: ", 1 - (total/10))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print("-----------------------------------------------------------------------------------------------------\n")

#Agora, usando nosso terceiro classificador como "DecisionTreeClassifier"

from sklearn import tree
clf = tree.DecisionTreeClassifier()
training = clf.fit(features_train,labels_train)
predictions = clf.predict(features_test)
print("Previsão: ",predictions)
print("Precisão como DecisionTreeClassifier:",clf.score(features_test,labels_test))

print("-----------------------------------------------------------------------------------------------------\n")

#Agora, usando nosso quarto classificador como "KNeighborsClassifier"

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
training = clf.fit(features_train,labels_train)
predictions = clf.predict(features_test)
print("Previsão: ",predictions)
print("Precisão como KNeighborsClassifier:",clf.score(features_test,labels_test))

print("-----------------------------------------------------------------------------------------------------\n")

#Podemos tentar usar outro classificador - "BaggingClassifier" Um classificador do Bagging é um
# meta-estimador conjunto que se encaixa classificadores base cada um em subconjuntos aleatórios do original
# conjunto de dados e, em seguida, agregar suas previsões individuais (por votação ou por média) para formar uma previsão final.

from sklearn.ensemble import BaggingClassifier
clf = BaggingClassifier(n_estimators=100, random_state=7)
boosted = clf.fit(features_train,labels_train)
prediction = clf.score(features_test,labels_test)
print("Precisão com BaggingClassifier:",prediction)

print("-----------------------------------------------------------------------------------------------------\n")

#Agora, podemos tentar prever a precisão usando o RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=7)
boosted = clf.fit(features_train,labels_train)
prediction = clf.score(features_test,labels_test)
print("Precisão usando o RandomForestClassifier:",prediction)

print("-----------------------------------------------------------------------------------------------------\n")

#Então, com isso, chegamos a saber que o SVM com o kernel 'rbf' supera outros algoritmos,
# além disso podemos aumentar a precisão ajustando o hiperparâmetro dos algoritmos usados

from sklearn import svm
clf = svm.SVC(kernel='rbf',C=10,gamma='auto')
training = clf.fit(features_train,labels_train)
predictions = clf.predict(features_test)
print("Previsão: ",predictions)
print("Precisão com rbf aumentando a precisão e ajustando o hiperparâmetro dos algoritmos usados:",clf.score(features_test,labels_test))

print("-----------------------------------------------------------------------------------------------------\n")



from sklearn.datasets import load_wine
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.model_selection import GridSearchCV

from tabulate import tabulate
import pandas as pd
from sklearn.model_selection import KFold

def make_meshgrid(X0, X1, h):
   
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

wine = load_wine()
X = wine.data[:,:2]
y = wine.target

X_initial_train, X_test, y_initial_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1) #divido inizialmente il dataset in train e test, dove il train contiene train e validation

X_train, X_validation, y_train, y_validation = train_test_split(X_initial_train, y_initial_train, test_size = 0.28, random_state = 1) #ora divido il train in train + validation

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                                               
c_params = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
Gamma_values = [10**(-7), 10**(-5), 10**(-3), 10**(-1), 0.5, 1, 10, 100]

h = 0.02

'''
-------------------------------------------------------------------------LINEAR AND RBF (WITHOUT TUNING OF GAMMA) SVM---------------------------------------------------------------------------
'''

kernels = ('linear', 'rbf')

for k in kernels:
    figure, axes = plt.subplots(2, 4)
    plt.subplots_adjust(wspace = 1, hspace = 1)
    accuracy = np.zeros(len(c_params))
    accuracy_best = -1
    C_Best = -1 #c associato al massimo valore di accuratezza da applicare al test set
    i = 0
    print("3-Class classification on Training Set with " +k + " SVM")
    for C_param, ax in zip(c_params, axes.flatten()):
        lin_svc = svm.SVC(kernel = k, C = C_param, gamma = 'auto')
        lin_svc.fit(X_train, y_train)
        
        #Evaluation of the accuracy
        accuracy[i] = lin_svc.fit(X_train, y_train).score(X_validation, y_validation)
        
        #calcolo del valore di C per cui l'accuratezza è massima
        if(accuracy[i] > accuracy_best):
            accuracy_best = accuracy[i]
            C_Best = C_param
        i += 1
        
        #stampa dei dati del training set   3-Class classification on Training Set with Linear SVM
        X0, X1 = X_train[:,0], X_train[:,1]
        xx, yy = make_meshgrid(X0, X1, h)
        
        Z = lin_svc.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.pcolormesh(xx, yy, Z, cmap = cmap_light)
        
        ax.scatter(X0, X1, c = y_train, cmap = cmap_bold, edgecolor = 'k', s = 20)
        ax.set_xlabel('Alcohol')
        ax.set_ylabel('Malic acid')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        title = "C = " + str(C_param)
        ax.set_title(title)
        ax.grid()
    plt.show()
        
    #plot dell'accuratezza
    print("Scores on validation:")
    print(accuracy)    
    print("max accuracy on validation: %.2f%%" %(100 * accuracy_best))
    plt.figure()
    plt.plot(c_params, accuracy)
    plt.semilogx()
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.grid()
    plt.title("Accuracy on validation with " + k + " kernel")
    plt.axhline(y = accuracy.max(), color = 'r', linestyle = '--', label = 'Maximum score on validation set')
    plt.legend(loc = "best")
    plt.show()
    
    # Evaluation on test set
    lin_svm_best = svm.SVC(kernel = k, C = C_Best, gamma = 'auto')
    lin_svm_best.fit(X_train, y_train)
    
    X0, X1 = X_test[:,0], X_test[:,1]
    xx, yy = make_meshgrid(X0, X1, h)
    print("accuracy on test set: %.2f%%" %(100 * lin_svm_best.fit(X_train, y_train).score(X_test, y_test)))
    Z = lin_svm_best.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)    
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)    
    plt.scatter(X0, X1, c = y_test, cmap = cmap_bold, edgecolor = 'k', s = 20)
    plt.xlabel('Alcohol')
    plt.ylabel('Malic acid')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    title = "(C = " + str(C_Best) + ")"
    plt.title("3-Class classification on Test Set with " + k + " SVM " + title)
    plt.grid()
    plt.show()
'''
#-------------------------------------------------------------------------RBF KERNEL SVM (TUNING OF GAMMA)---------------------------------------------------------------------------
'''

accuracy_C_Gamma = np.zeros(len(c_params) * len(Gamma_values))
accuracy_best_C_Gamma = -1
C_Best2 = -1 #c associato al massimo valore di accuratezza da applicare al test set
Gamma_best = -1
i = 0
for C_param in c_params:
    
    for Gamma in Gamma_values:
        
        rbf_svc = svm.SVC(kernel = 'rbf', gamma = Gamma, C = C_param)
        rbf_svc.fit(X_train, y_train)
    
        #Evaluation of the accuracy
        accuracy_C_Gamma[i] = rbf_svc.fit(X_train, y_train).score(X_validation, y_validation)
        
        #calcolo del valore di C per cui l'accuratezza è massima
        if(accuracy_C_Gamma[i] > accuracy_best_C_Gamma):
            accuracy_best_C_Gamma = accuracy_C_Gamma[i]
            C_Best2 = C_param
            Gamma_best = Gamma
        i += 1
        
#per ogni C e Gamma stampo la media delle accuratezze (somma su tutti gli split del validation fratto il numero di split)
df = pd.DataFrame(np.reshape([round(item * 100, 2) for item in accuracy_C_Gamma], (7, 8)), index = ([str(item) for item in c_params]), columns = ([str(item) for item in Gamma_values]))
print()
print("Rows = C; Columns = Gamma; Values are in %")
print(tabulate(df, headers = 'keys', tablefmt = 'psql', numalign = "center"))
title = "{C = " + str(C_Best2) + ", Gamma = " + str(Gamma_best) +"}"
print("Best parameters: " + title)
print("Best score: %.2f%%" % (accuracy_best_C_Gamma * 100))

plt.matshow(df)
plt.gca().xaxis.tick_bottom()
plt.xlabel('Gamma')
plt.ylabel('C')
cbar = plt.colorbar()
cbar.set_label('Score on validation set')
plt.yticks(np.arange(len(c_params )), c_params)
plt.xticks(np.arange(len(Gamma_values)), Gamma_values )      
plt.show()      
     

#train della svm con i migliori C e Gamma
rbf_svm_best_C_Gamma = svm.SVC(kernel = 'rbf', gamma = Gamma_best, C = C_Best2)
rbf_svm_best_C_Gamma.fit(X_train, y_train)

#stampa dei dati del training set

X0, X1 = X_train[:,0], X_train[:,1]
xx, yy = make_meshgrid(X0, X1, h)

Z = rbf_svm_best_C_Gamma.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = cmap_light)

plt.scatter(X0, X1, c = y_train, cmap = cmap_bold, edgecolor = 'k', s = 20)
plt.xlabel('Alcohol')
plt.ylabel('Malic acid')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification on Training Set with RBF SVM " + title)
plt.grid()
plt.show()

print("Accuracy on test set: %.2f%%" %(100 * rbf_svm_best_C_Gamma.fit(X_train, y_train).score(X_test, y_test)))

#print("max accuracy on training: %f" %accuracy_best_C_Gamma)

X0, X1 = X_test[:,0], X_test[:,1]
xx, yy = make_meshgrid(X0, X1, h)

Z = rbf_svm_best_C_Gamma.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
    
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
    
plt.scatter(X0, X1, c = y_test, cmap = cmap_bold, edgecolor = 'k', s = 20)
plt.xlabel('Alcohol')
plt.ylabel('Malic acid')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification on Test Set with RBF SVM " + title)
plt.grid()
plt.show()

'''
#------------------------------------------------------------------------- 5 FOLD CROSS VALIDATION WITH RBF KERNEL SVM ---------------------------------------------------------------------------
'''
#Merge the training and validation split
np.concatenate((X_train, X_validation), axis=None)
np.concatenate((y_train, y_validation), axis=None)

parameters  = {'gamma' : Gamma_values, 'C' : c_params}

rbf_svc = svm.SVC(kernel = 'rbf')
kfolds = KFold(5)
rbf_svc_clf = GridSearchCV(rbf_svc, parameters, cv = kfolds.split(X_train, y_train), iid = True)
rbf_svc_clf.fit(X_train, y_train)

#per ogni C e Gamma stampo la media delle accuratezze (somma su tutti gli split del validation fratto il numero di split)
df = pd.DataFrame(np.reshape([round(item*100, 2) for item in rbf_svc_clf.cv_results_['mean_test_score']], (7,8)), index=([str(item) for item in c_params]), columns=([str(item) for item in Gamma_values]))
print()
print("Rows = C; Columns = Gamma; Values are in %")
print(tabulate(df, headers = 'keys', tablefmt = 'psql', numalign = "center"))
print("Best parameters: %s" % (rbf_svc_clf.best_params_))
print("Best score: %.2f%%" % (rbf_svc_clf.best_score_ * 100))

plt.matshow(df)
plt.gca().xaxis.tick_bottom()
plt.xlabel('Gamma')
plt.ylabel('C')
cbar = plt.colorbar()
cbar.set_label('Score on validation set')
plt.yticks(np.arange(len(c_params)), c_params)
plt.xticks(np.arange(len(Gamma_values )),Gamma_values)
plt.show()

C_Best2    = rbf_svc_clf.best_params_['C']
Gamma_best = rbf_svc_clf.best_params_['gamma']

#stampa dei dati del training set

X0, X1 = X_train[:,0], X_train[:,1]
xx, yy = make_meshgrid(X0, X1, h)
Z = rbf_svc_clf.best_estimator_.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = cmap_light)

plt.scatter(X0, X1, c = y_train, cmap = cmap_bold, edgecolor = 'k', s = 20)
plt.xlabel('Alcohol')
plt.ylabel('Malic acid')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
values = "C = " + str(C_Best2) + ", Gamma =" + str(Gamma_best)
plt.title("3-Class classification on Training Set with RBF SVM (" + values + ")")
plt.grid()
plt.show()

#stampa dei dati del test set

accuracy_best_C_Gamma = rbf_svc_clf.best_estimator_.score(X_test, y_test)
print("Accuracy on test set: %.2f%%" %(100 * accuracy_best_C_Gamma))

X0, X1 = X_test[:,0], X_test[:,1]
xx, yy = make_meshgrid(X0, X1, h)

Z = rbf_svc_clf.best_estimator_.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
    
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
    
plt.scatter(X0, X1, c = y_test, cmap = cmap_bold, edgecolor = 'k', s = 20)
plt.xlabel('Alcohol')
plt.ylabel('Malic acid')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
values = "C = " + str(C_Best2) + ", Gamma =" + str(Gamma_best)
plt.title("3-Class classification on Test Set with RBF SVM (" + values + ")")
plt.grid()
plt.show()
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import neighbors

def make_meshgrid(X0, X1, h):
  
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    return xx, yy

wine = load_wine()
X = wine.data[:,:2] #per ogni riga prendo 2 colonne
y = wine.target

X_initial_train, X_test, y_initial_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1) #divido inizialmente il dataset in train e test, dove il train contiene train e validation

X_train, X_validation, y_train, y_validation = train_test_split(X_initial_train, y_initial_train, test_size = 0.28, random_state = 1) #ora divido il train in train + validation

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                            
n_neighbors = [1, 3, 5, 7]
h = 0.02
K_Best = -1
W_Best = ''
accuracy_best = -1

weights = ('uniform', 'distance')
                            
for w in weights:
    figure, axes = plt.subplots(2, 2)
    plt.subplots_adjust(wspace = 1, hspace = 1)
    accuracy = np.zeros(len(n_neighbors))
    acc_uni_dist = -1
    i = 0
    for k, ax in zip(n_neighbors, axes.flatten()):
        # we create an instance of Neighbours Classifier and fit the data.
        clf_knn = neighbors.KNeighborsClassifier(k, weights = w)
        clf_knn.fit(X_train, y_train)
        
        accuracy[i] = clf_knn.fit(X_train, y_train).score(X_validation, y_validation)
     
        #find max value of accuracy  
        if(accuracy[i] > accuracy_best):
            accuracy_best = accuracy[i]
            K_Best = k
            if (w == 'uniform'):
                W_Best = 'uniform'
            else:
                W_Best = 'distance'
        #calcolo locale dell'accuratezza migliore (per ogni peso)
        if(accuracy[i] > acc_uni_dist):
            acc_uni_dist = accuracy[i]
        i += 1
        
         # Plot the decision boundary
        X0, X1 = X_train[:,0], X_train[:,1]
        xx, yy = make_meshgrid(X0, X1, h)
        
        Z = clf_knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.pcolormesh(xx, yy, Z, cmap = cmap_light)
        
        ax.scatter(X0, X1, c = y_train, cmap = cmap_bold, edgecolor = 'k', s = 20)
        ax.set_xlabel('Alcohol')
        ax.set_ylabel('Malic acid')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        title = "K = " + str(k)
        ax.set_title(title)
        ax.grid()
      

    plt.show()

    print("Scores on validation (" + w + "):")
    print(accuracy)
 
    print("max accuracy on validation: %.2f%%" %(100 * acc_uni_dist))
    plt.figure()
    plt.plot(n_neighbors, accuracy)
    plt.xlabel('k')
    plt.ylabel("accuracy")
    plt.grid()
    plt.title("Accuracy on validation with " + w + " weight")
    plt.axhline(y = accuracy.max(), color = 'r', linestyle = '--', label = 'Maximum score on validation set')
    plt.legend(loc = "best")
    plt.show()

# Evaluation on test set
clf_knn_best = neighbors.KNeighborsClassifier(K_Best, weights = W_Best)
clf_knn_best.fit(X_train, y_train)

X0, X1 = X_test[:,0], X_test[:,1]
xx, yy = make_meshgrid(X0, X1, h)
print("accuracy on test set: %.2f%%" %(100 * clf_knn_best.fit(X_train, y_train).score(X_test, y_test)))
Z = clf_knn_best.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)    
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = cmap_light)    
plt.scatter(X0, X1, c = y_test, cmap = cmap_bold, edgecolor = 'k', s = 20)
plt.xlabel('Alcohol')
plt.ylabel('Malic acid')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
title = "(K = " + str(K_Best) + " weights = " + W_Best + ")"
plt.title("3-Class classification on Test Set with KNN " + title)
plt.grid()
plt.show()
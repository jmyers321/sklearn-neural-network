from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV


### DATA ###
seed = 1
X, y = make_classification(n_samples=4000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2, random_state=seed)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=seed)


### ALGORITHM ###


parameters = {'hidden_layer_sizes': [(5), (10), (10,2), (50)]}
algo = GridSearchCV(MLPClassifier(max_iter = 1000), parameters, verbose = 4)


### TRAINING ###

algo.fit(X_train, y_train)

### PREDICTION/TESTING ###

y_pred = algo.predict(X_test)
acc1 = accuracy_score(y_pred,y_test)
acc2 = (y_test == y_pred).sum()/len(y_test)
np.testing.assert_almost_equal(acc1,acc2, decimal=2)
print(acc1 == acc2)

plt.scatter(X_train[:,0], X_train[:,1], c=y_train, s=0.8,label="Training")
plt.scatter(X_test[:,0], X_test[:,1], marker='x', c=y_pred, label="Testing")
plt.axis('equal')
plt.legend()

plt.title(f"Binary Classifier With Accuracy {(y_test == y_pred).sum()/len(y_test)*100}%")
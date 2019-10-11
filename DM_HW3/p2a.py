import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics



Transactions_sample= pd.read_csv("svmdata.csv")
z_train, z_test \
    = train_test_split(Transactions_sample, test_size=0.3, random_state=1)
z_train, z_val \
    = train_test_split(Transactions_sample, test_size=0.3, random_state=1)

def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


X = z_train[['First']+['Second']]
X_test = z_test[['First']+['Second']]
y = z_train[['Third']].values.ravel()
y_test = z_test[['Third']].values.ravel()
plt.scatter(X['First'], X['Second'], c=y, cmap=plt.cm.Paired)
plt.show()
error = []
ax = plt.gca()
accf = -1;
C = -1;
for C in np.arange(0.1, 1,0.1):
    clf = svm.SVC(kernel='linear', C=C)
    model = clf.fit(X, y)
    y_pred = clf.predict(X_test)
    error.append(np.mean(y_pred != y_test))
    acc = metrics.accuracy_score(y_test, y_pred)
    if(acc > accf):
        accf = acc
        cfin = C
    del clf,model, y_pred,acc

plt.plot(np.arange(0.1, 1,0.1), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.show()

print "the best C is",cfin,"and the accuracy of it is",accf
plt.scatter(X['First'], X['Second'], c=y, cmap=plt.cm.Paired)
clf = svm.SVC(kernel='linear', C=cfin)
model = clf.fit(X, y)
y_pred = clf.predict(X_test)
error.append(np.mean(y_pred != y_test))
plot_svc_decision_function(model)
plt.show()



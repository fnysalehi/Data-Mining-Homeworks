import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

path = "venv/US Presidential Data.csv"

# Assign colum names to the dataset
#y  = np.array(['Win/Loss',	'Optimism',	'Pessimism', 'PastUsed', 'FutureUsed',	'PresentUsed',	'OwnPartyCount',	'OppPartyCount'	, 'NumericContent',	'Extra',	'Emoti',	'Agree',	'Consc',	'Openn'])

# Read dataset to pandas dataframe
dataset = pd.read_csv(path)


#print(dataset)

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:,0]
#print(X)
#print(y)
#X = X.tranpose()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#classifier = KNeighborsClassifier(n_neighbors=1)
#classifier.fit(X_train, y_train)

#y_pred = classifier.predict(X_test)

#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
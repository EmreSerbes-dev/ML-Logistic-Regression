import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression #For regression
from sklearn.model_selection import train_test_split #For split data
from sklearn.preprocessing import StandardScaler #For feature scaling
from sklearn.metrics import confusion_matrix, accuracy_score #For condusing matrix
from matplotlib.colors import ListedColormap  #For graphic

#For datas
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Split the datas
xTrain, xTest, yTrain,yTest = train_test_split(x, y, test_size = 0.25, random_state = 0)

#Feature scaling
sc = StandardScaler()
xTrain = sc.fit_transform(xTrain)
xTest = sc.fit_transform(xTest)

#Regression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xTrain, yTrain)

#Predict new result
z = classifier.predict([[1659, 87000]]) #New purchased result
# print(z)

#Predict test results
yPred = classifier.predict(xTest)
np.set_printoptions(precision = 2)
print(f"{np.concatenate((yPred.reshape(len(yPred), 1), yTest.reshape(len(yTest), 1)), 1)} \n")

#Confusing matrix
cm = confusion_matrix(yTest, yPred)
print(cm)
accuracy_score(yTest, yPred)

#Graphic results
xSet, ySet = sc.inverse_transform(xTrain), yTrain
x1, x2 = np.meshgrid(np.arange(start = xSet[:, 0].min() - 10, stop = xSet[:, 0].max() + 10, step = 0.25),
                     np.arange(start = xSet[:, 1].min() - 1000, stop = xSet[:, 1].max() + 1000, step = 0.25))
plt.contourf(x1, x2, classifier.predict(sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(ySet)):
    plt.scatter(xSet[ySet == j, 0], xSet[ySet == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

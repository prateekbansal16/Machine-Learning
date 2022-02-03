import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
dataset = load_iris()

df = pd.DataFrame(data= np.c_[dataset['data'], dataset['target']],
                     columns= dataset['feature_names'] + ['target'])

X=dataset.data
y=dataset.target

plt.scatter(X[y == 0, 0] , X[y == 0, 1], c = "r", label = "Setosa")
plt.scatter(X[y == 1, 0] , X[y == 1, 1], c = "g", label = "Versicolor")
plt.scatter(X[y == 2, 0] , X[y == 2, 1], c = "b", label = "Verginica")
plt.legend()
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Analysis on the Iris dataset')
plt.show()

plt.scatter(X[y == 0, 1] , X[y == 0, 2], c = "r", label = "Setosa")
plt.scatter(X[y == 1, 1] , X[y == 1, 2], c = "g", label = "Versicolor")
plt.scatter(X[y == 2, 1] , X[y == 2, 2], c = "b", label = "Verginica")
plt.legend()
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Analysis on the Iris dataset')
plt.show()


plt.scatter(X[y == 0, 2] , X[y == 0, 3], c = "r", label = "Setosa")
plt.scatter(X[y == 1, 2] , X[y == 1, 3], c = "g", label = "Versicolor")
plt.scatter(X[y == 2, 2] , X[y == 2, 3], c = "b", label = "Verginica")
plt.legend()
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Analysis on the Iris dataset')
plt.show()

corr= df.corr() # finding the co -relation

import seaborn as sn

fig, ax = plt.subplots(figsize=(5,5))
sn.heatmap(corr, annot=True, ax=ax)
sn.show()

from sklearn.model_selection import train_test_split
#training data allocation - 70
#test data allocation - 30
x_train, x_test, y_train, y_test = train_test_split (X, y, test_size = 0.30 )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

print("Accuracy: " , model.score(x_train, y_train)*100)

# make a prediction
Xnew = [[5.4,3.9,1.7,0.4]]
ynew = model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

if model.predict(Xnew) == 2:
    print("Flower is Verginica")
elif model.predict(Xnew) == 1:
    print("Flower is Versicolor")
elif model.predict(Xnew) == 0:
    print("Flower is Setosa")
    




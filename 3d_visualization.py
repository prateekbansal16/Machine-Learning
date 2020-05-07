from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('Dataset/StudentsPerformance.csv')

X = dataset.iloc[ : ,7].values
Y = dataset.iloc[ : ,6].values
Z = dataset.iloc[ : ,5].values

 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,Z, c='red', s=30)

ax.view_init(30, 185)
ax.set_xlabel('Wrting Score')
ax.set_ylabel('Reading Score')
ax.set_zlabel('Math Score')
plt.show()



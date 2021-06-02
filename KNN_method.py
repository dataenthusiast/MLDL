import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
# Load Data
df = pd.read_csv('well_data_with_facies.csv')
# Data train
well_train = df.loc[df['Well Name']=='STUART']
xtrain = well_train.loc[:, 'GR':'RELPOS']
ytrain = well_train.loc[:, 'Facies']
# Data Test
well_test  = df.loc[df['Well Name']!='STUART']
xtest = well_test.loc[:, 'GR':'RELPOS']
ytest = well_test.loc[:, 'Facies']
# Scaler
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.fit_transform(xtest)
# cross validation
neighbors = list(range(1, 100, 2))
cv_score = []
for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, xtrain, ytrain, cv=10, scoring='accuracy')
    cv_score.append(scores.mean())
MSE = [1-x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print('Optimum N = ', optimal_k)
knn = KNeighborsClassifier(n_neighbors= optimal_k)
knn.fit(xtrain, ytrain)
# Predict Data
y_predict = knn.predict(xtest)
print('Accuracy = ', metrics.accuracy_score(ytest, y_predict))
# Create Confusion Matrix
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(ytest, y_predict)
print(conf)
depth = well_train.loc[:,'Depth']
# Coefficient Correlation
import numpy as np
print(np.corrcoef(y_predict,ytest))
plt.figure()
# Plot Predict and Original Data
plt.subplot(1, 3, 1)
plt.title('Test')
cluster=np.repeat(np.expand_dims(ytest.values,1), 1, 1)
plt.imshow(cluster, interpolation='none', aspect='auto',vmin=1,vmax=9, extent=[0,1,max(depth),min(depth)])
plt.colorbar()
plt.subplot(1, 3, 3)
cluster=np.repeat(np.expand_dims(y_predict,1), 1, 1)
a = plt.imshow(cluster, interpolation='none', aspect='auto',vmin=1,vmax=9, extent=[0,1,max(depth),min(depth)])
plt.title('Predict')
plt.colorbar(a)
# Plot Error Value
plt.figure()
plt.plot(neighbors, MSE)
plt.title('RMS Error')
plt.show()
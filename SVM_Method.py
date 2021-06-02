import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
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
clf = svm.SVC(kernel='linear')
clf.fit(xtrain, ytrain)
y_predict = clf.predict(xtest)
print("Accuracy", metrics.accuracy_score(ytest, y_predict))
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
plt.show()
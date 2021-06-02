import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

df = pd.read_csv('well_data_with_facies.csv')
stuart = df.loc[df['Well Name']=='STUART']
crawford = df.loc[df['Well Name'] =='CRAWFORD']
x = stuart.loc[:, 'GR':'RELPOS']
y = stuart.loc[:, 'Facies']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test =train_test_split(x_scaled, y,test_size=0.3,
                                                   random_state=1, stratify= y)
# cross validation
nestimator = list(range(1, 50, 1))
cv_score = []
for i in nestimator:
    clf = RandomForestClassifier(n_estimators = i)
    scores = cross_val_score(clf, x_scaled, y, cv=10, scoring='accuracy')
    cv_score.append(scores.mean())
MSE = [1-x for x in cv_score]
optimal_n = nestimator[MSE.index(min(MSE))]
print(optimal_n)

clf = RandomForestClassifier(n_estimators= optimal_n, random_state=40)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print('Accuracy', metrics.accuracy_score(y_test, y_predict))

newdata = crawford.loc[:, 'GR':'RELPOS']
true_fc = crawford.loc[:, 'Facies']
nd_predict = clf.predict(newdata)

plt.figure()
ims = list(zip(crawford['NM_M'], true_fc))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(ims, aspect='auto')
plt.xlim([0.5, 1.5])
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title('Predict')
d2 = list(zip(crawford['NM_M'], nd_predict))
plt.imshow(d2, aspect='auto')
plt.xlim([0.5, 1.5])
plt.colorbar()

plt.figure()
mat = metrics.confusion_matrix(true_fc, nd_predict)
a = sns.heatmap(mat, annot=True, fmt='d', cbar=False)
plt.xlabel('True')
plt.ylabel('Predict')
bottom, top = a.get_ylim()
a.set_ylim(bottom+0.5, top-0.5)

plt.show()

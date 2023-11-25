import sklearn as sk
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import pickle

df = pd.read_csv('train.csv')


set = ()

# read each line in dataframe and add distinct letters in values to set
for index, row in df.iterrows():
    for i in row['Sequence']:
        if i not in set:
            set = set + (i,)
    
# sort the set
set = sorted(set)

print(set)

# unique letters in the set
print(len(set))

X = pd.DataFrame(columns=['ID'] + list(set))

print(X)


X.drop('ID', axis=1, inplace=True)

for index, row in df.iterrows():
    X.loc[index] = [row['Sequence'].count(i)/len(row['Sequence']) for i in set]

X.head()

Y = df['Label']

Y.head()

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)

X_resampled, Y_resampled = ros.fit_resample(X, Y)

# number of samples in each class after oversampling
Y_resampled.value_counts()


X = X_resampled
Y = Y_resampled

# each row of each column in X
total_0 = [0] * 20
total_1 = [0] * 20

k = 0

for i in X.columns:
    
    for j in range(len(X)):
        if Y[j] == 1:
            total_1[k] += X[i][j]
        else:
            total_0[k] += X[i][j]
    k += 1

print(total_1)
print(total_0)

min_1 = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000]
min_0 = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000]

k = 0

for i in X.columns:
    for j in range(len(X)):
        if Y[j] == 1:
            if X[i][j] < min_1[k] and X[i][j] != 0:
                min_1[k] = X[i][j]
        else:
            if X[i][j] < min_0[k] and X[i][j] != 0:
                min_0[k] = X[i][j]
    k += 1
print(min_1)
print(min_0)

max_1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
max_0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

pp = 0
for i in X.columns:
    for j in range(len(X)):
        if Y[j] == 1:
            if X[i][j] > max_1[pp]:
                max_1[pp] = X[i][j]
        else:
            if X[i][j] > max_0[pp]:
                max_0[pp] = X[i][j]
    pp += 1

print(max_1)
print(max_0)
    
k = 0
for i in X.columns:
    for j in range(len(X)):
        if Y[j] == 1:
            if X[i][j] == 0:
                X[i][j] = ((total_1[k]/1127) - min_1[k])/(max_1[k] - min_1[k])
        else:
            if X[i][j] == 0:
                X[i][j] = ((total_0[k]/1127) - min_0[k])/(max_0[k] - min_0[k])
    k += 1

# X.head(100)

from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

model = keras.Sequential(
    [
        keras.Input(shape=(20,)),
        
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dropout(0.3),
        # keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(100, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(0.01))
    ]
)

# dropout layer
# keras.layers.Dropout(0.3)


# train the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_X, train_Y, epochs=1200, batch_size=5)

# plot epochs vs loss

# store model pickle file
import pickle

pickle.dump(model, open('model.pkl', 'wb'))


import matplotlib.pyplot as plt

plt.plot(model.history.history['f'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# AUC ROC curve

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(test_Y, model.predict(test_X, batch_size=10))

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()

# # predict the model
# model.predict(test_X, batch_size=10)

# change values predicted to binary values 0 or 1
# # model.predict_classes(test_X, batch_size=10)
# model.predict(test_X, batch_size=10)

# convert values to binary


# check the accuracy of the model
from sklearn.metrics import accuracy_score

accuracy_score(test_Y, model.predict(test_X, batch_size=10).round().astype(int))
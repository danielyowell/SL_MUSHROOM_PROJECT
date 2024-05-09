import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder

M = fetch_openml(data_id=24)
x, y = M.data, M.target

#! Convert the data to a pandas DataFrame
M_features_df = pd.DataFrame(x, columns=M.feature_names)
M_targets_df = pd.DataFrame(y, columns=['class'])

M_everything_df = pd.concat([M_features_df, M_targets_df], axis=1)
# print(M_everything_df.head())

'''
CONVERT TO NUMERICAL DATA
'''
from mapping import mapping
for column, mp in mapping.items():
    M_everything_df[column] = M_everything_df[column].replace(mp)
# print(M_everything_df.head())

le = LabelEncoder()
M_everything_df_encoded = M_everything_df.apply(le.fit_transform)

'''
CLEAN DATA; VERIFY WITH STANDARD DEVIATION AND MEAN
'''
# veil-type has std = mean = 0 
# remove veil-type column
M_everything_df_encoded = M_everything_df_encoded.drop(columns='veil-type')

# change to 1 to see std/mean outputs
if(0):
    print("standard deviations:")
    std = M_everything_df_encoded.std()
    print(std)
    print()
    print("mean:")
    mean = M_everything_df_encoded.mean()
    print(mean)

'''
SPLIT DATA
'''
# Now split and use the encoded DataFrame
train, test = train_test_split(M_everything_df_encoded, test_size=.2, random_state=42)

X_train = train.iloc[:,:-1].values
Y_train = train.iloc[:,-1].values

X_test = test.iloc[:,:-1].values
Y_test = test.iloc[:,-1].values

##

import math

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)

import operator

def get_neighbors(train, test_row, num_neighbors):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def predict_classification(train, test_row, num_neighbors):
    # print("test_row:", test_row)
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    # print("prediction:", prediction)
    return prediction

train_set = train.values
test_set = test.values
num_neighbors = 5
predictions = []

counter = 0
print("length of test set:", len(test_set))
print("counter:", counter)
for test_row in test_set:
    output = predict_classification(train_set, test_row, num_neighbors)
    predictions.append(output)
    counter += 1
    print("counter:", counter)

print(predictions)

from sklearn.metrics import accuracy_score, classification_report
# Assuming you have a list of predictions and true labels
true_labels = [test_row[-1] for test_row in test_set]
accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)
report = classification_report(true_labels, predictions)
print("Classification Report:\n", report)

if(0):
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, Y_train)

    y_pred = knn.predict(X_test)

    from sklearn.metrics import accuracy_score, classification_report

    print("Accuracy:", accuracy_score(Y_test, y_pred))
    print("Classification Report:\n", classification_report(Y_test, y_pred))
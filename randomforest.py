import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from numpy.random import choice, seed
from sklearn.metrics import accuracy_score

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DTClassifier(BaseEstimator):
    def __init__(self, min_samples_split=2, max_depth=2, criterion='gini', max_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.root = None
        self.feature_importances_ = None
        self.feature_indices_ = None

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])
        if self.max_features is not None:
            self.feature_indices_ = choice(range(X.shape[1]), self.max_features, replace=False)
            X = X[:, self.feature_indices_]
        else:
            self.feature_indices_ = range(X.shape[1])
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset, 0)
        self.feature_importances_ /= np.sum(self.feature_importances_)
        return self
    
    def predict(self, X):
        if self.max_features is not None:
            X = X[:, self.feature_indices_]
        return np.array([self._make_prediction(x, self.root) for x in X])

    def _make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._make_prediction(x, tree.left)
        else:
            return self._make_prediction(x, tree.right)

    def build_tree(self, dataset, depth):
        num_samples, num_features = dataset.shape[0], dataset.shape[1] - 1
        if num_samples < self.min_samples_split or depth == self.max_depth:
            return TreeNode(value=self.calculate_leaf_value(dataset[:, -1]))

        best_split = self.get_best_split(dataset, num_features)
        if not best_split or best_split['info_gain'] == 0:
            return TreeNode(value=self.calculate_leaf_value(dataset[:, -1]))

        self.feature_importances_[best_split['feature_index']] += best_split['info_gain']

        left_subtree = self.build_tree(best_split['dataset_left'], depth + 1)
        right_subtree = self.build_tree(best_split['dataset_right'], depth + 1)

        return TreeNode(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree, best_split['info_gain'])

    def get_best_split(self, dataset, num_features):
        best_split = None
        max_info_gain = -float('inf')
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) == 0 or len(dataset_right) == 0:
                    continue

                info_gain = self.information_gain(dataset, dataset_left, dataset_right)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'info_gain': info_gain,
                        'dataset_left': dataset_left,
                        'dataset_right': dataset_right
                    }
        return best_split if best_split and max_info_gain > 0 else None

    def split(self, dataset, feature_index, threshold):
        left_idx = dataset[:, feature_index] <= threshold
        right_idx = dataset[:, feature_index] > threshold
        return dataset[left_idx], dataset[right_idx]

    def information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        gain = self.gini(parent) - (weight_left * self.gini(left_child) + weight_right * self.gini(right_child))
        return gain

    def gini(self, subset):
        _, counts = np.unique(subset[:, -1], return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)

    def calculate_leaf_value(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

class RandomForestClassifierCustom(BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 criterion='gini', max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        seed(self.random_state)
        self.trees = []
        for _ in range(self.n_estimators):
            indices = choice(len(y), len(y), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = DTClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        self.aggregate_feature_importances(X.shape[1])
        return self

    def aggregate_feature_importances(self, num_features):
        importances = np.zeros(num_features)
        for tree in self.trees:
            if tree.feature_indices_ is not None:
                importances[tree.feature_indices_] += tree.feature_importances_
        self.feature_importances_ = importances / np.sum(importances)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x, minlength=2).argmax(), axis=0, arr=predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# Load dataset
data = pd.read_csv("mushrooms.csv")

# Convert categorical variables to numerical format using one-hot encoding
data_encoded = pd.get_dummies(data)
data_encoded.drop(['class_e'], axis=1, inplace=True)  # Assuming 'class_p' is the target variable

# Splitting Data
X = data_encoded.drop('class_p', axis=1).values
y = data_encoded['class_p'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train custom random forest classifier
rf_custom = RandomForestClassifierCustom(n_estimators=100, max_depth=None, min_samples_split=2,
                                         criterion='gini', max_features=None, random_state=42)
rf_custom.fit(X_train, y_train)

# Evaluate model performance on test data
y_pred_train = rf_custom.predict(X_train)
y_pred_test = rf_custom.predict(X_test)

# Model performance evaluation with cross-validation
cv_scores = cross_val_score(rf_custom, X_train, y_train, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation: {np.std(cv_scores):.4f}\n")

# Output results
print("TRAINING RESULTS:")
print(f'Accuracy Score: {accuracy_score(y_train, y_pred_train):.4f}')
print('Classification Report:\n', classification_report(y_train, y_pred_train))
print('Confusion Matrix:\n', confusion_matrix(y_train, y_pred_train), '\n')

print("TEST RESULTS:")
print(f'Accuracy Score: {accuracy_score(y_test, y_pred_test):.4f}')
print('Classification Report:\n', classification_report(y_test, y_pred_test))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_test))

# Output feature importances
features = data_encoded.drop('class_p', axis=1).columns
importances_df = pd.DataFrame({'Feature': features, 'Importance': rf_custom.feature_importances_})
importances_df = importances_df


# Display the top 10 most important features
print(importances_df.head(10))

# Plotting feature importances
plt.figure(figsize=(12, 6))
plt.title('Random Forest Feature Importance')
plt.barh(importances_df['Feature'][:10], importances_df['Importance'][:10], color='b')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important at the top
plt.show()
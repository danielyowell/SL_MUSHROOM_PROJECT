import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
#from sklearn.tree import DecisionTreeClassifier, #Originally used for baseline performance
#from sklearn.tree import plot_tree, #For testing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz
import graphviz
from sklearn.base import BaseEstimator

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DTClassifier(BaseEstimator):
    """ A simple decision tree classifier implementing Gini impurity for information gain. """
    def __init__(self, min_samples_split=2, max_depth=2, criterion='gini'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None
        self.feature_importances_ = None  # Initialize feature_importances_

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])  # Reset feature_importances_ on fit
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset, 0)
        self.feature_importances_ /= np.sum(self.feature_importances_)  # Normalize feature importances
        return self

    def predict(self, X):
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

# Load dataset
data = pd.read_csv("mushrooms.csv")

# One hot encoding
data_encoded = pd.get_dummies(data)
data_encoded.drop(['class_e'], axis=1, inplace=True)

# Splitting Data
X = data_encoded.drop('class_p', axis=1).values  # Assuming 'class_p' is the target variable column
y = data_encoded['class_p'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# training
dt_classifier = DTClassifier(max_depth=10, min_samples_split=5, criterion='gini')
dt_classifier.fit(X_train, y_train)
y_pred_test = dt_classifier.predict(X_test)

# Calculate accuracy
def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

print("TEST RESULTS:")
accuracy = calculate_accuracy(y_test, y_pred_test)
print(f'Accuracy Score: {accuracy:.4f}')
print('Classification Report:\n', classification_report(y_test, y_pred_test))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_test))

# Feature Importance Analysis
def calculate_feature_importance(tree, num_features):
    importance = np.zeros(num_features)
    def traverse(node):
        if node.value is None:
            importance[node.feature_index] += node.info_gain
            traverse(node.left)
            traverse(node.right)
    traverse(tree)
    return importance / importance.sum()

importances = calculate_feature_importance(dt_classifier.root, X_train.shape[1])
features = data_encoded.drop('class_p', axis=1).columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Recursive Feature Elimination (RFE) for feature selection
rfe = RFECV(estimator=DTClassifier(max_depth=10, min_samples_split=5, criterion='gini'), step=1, cv=5, scoring='accuracy')
rfe.fit(X_train, y_train)

# Apply RFE transformation to both training and testing data
features_rfe = data_encoded.drop('class_p', axis=1).columns[rfe.support_]  # Get the column names of the selected features
X_train_rfe = pd.DataFrame(rfe.transform(X_train), columns=features_rfe)
X_test_rfe = pd.DataFrame(rfe.transform(X_test), columns=features_rfe)
X_train_rfe.index = pd.RangeIndex(start=0, stop=len(y_train))

# Fit the model on the RFE-transformed training data
dt_classifier_rfe = DTClassifier(max_depth=10, min_samples_split=5, criterion='gini')
dt_classifier_rfe.fit(X_train_rfe.values, y_train)
dt_classifier_rfe.fit(X_train_rfe.values, y_train.values if isinstance(y_train, pd.Series) else y_train)


# Predict on the RFE-transformed training and test data
y_pred_train_rfe = dt_classifier_rfe.predict(X_train_rfe.values)
y_pred_test_rfe = dt_classifier_rfe.predict(X_test_rfe.values)

# Calculate accuracies for RFE model
accuracy_train_rfe = calculate_accuracy(y_train, y_pred_train_rfe)
accuracy_test_rfe = calculate_accuracy(y_test, y_pred_test_rfe)
print("RFE TRAINING RESULTS:")
print(f'Accuracy Score: {accuracy_train_rfe:.4f}')
print('Classification Report:\n', classification_report(y_train, y_pred_train_rfe))
print('Confusion Matrix:\n', confusion_matrix(y_train, y_pred_train_rfe))

print("RFE TESTING RESULTS:")
print(f'Accuracy Score: {accuracy_test_rfe:.4f}')
print('Classification Report:\n', classification_report(y_test, y_pred_test_rfe))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_test_rfe))

# Reduce highly correlated features
corr_matrix = pd.DataFrame(X_train, columns=data_encoded.drop('class_p', axis=1).columns).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_train_reduced = pd.DataFrame(X_train, columns=data_encoded.drop('class_p', axis=1).columns).drop(to_drop, axis=1)
X_test_reduced = pd.DataFrame(X_test, columns=data_encoded.drop('class_p', axis=1).columns).drop(to_drop, axis=1)

# Fit and evaluate the model on reduced feature set
dt_classifier_reduced = DTClassifier(max_depth=10, min_samples_split=5, criterion='gini')
dt_classifier_reduced.fit(X_train_reduced.values, y_train)

y_pred_train_reduced = dt_classifier_reduced.predict(X_train_reduced.values)
y_pred_test_reduced = dt_classifier_reduced.predict(X_test_reduced.values)

accuracy_train_reduced = calculate_accuracy(y_train, y_pred_train_reduced)
accuracy_test_reduced = calculate_accuracy(y_test, y_pred_test_reduced)
print("Reduced Correlation TRAINING RESULTS:")
print(f'Accuracy Score: {accuracy_train_reduced:.4f}')
print('Classification Report:\n', classification_report(y_train, y_pred_train_reduced))
print('Confusion Matrix:\n', confusion_matrix(y_train, y_pred_train_reduced))

print("Reduced Correlation TESTING RESULTS:")
print(f'Accuracy Score: {accuracy_test_reduced:.4f}')
print('Classification Report:\n', classification_report(y_test, y_pred_test_reduced))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_test_reduced))

# Plot the comparison of model performance before and after feature reduction
results_df = pd.DataFrame({
    'Model': ['Original', 'RFE', 'Reduced Correlation'],
    'Training Accuracy': [
        accuracy_score(y_train, dt_classifier.predict(X_train)),
        accuracy_train_rfe,
        accuracy_train_reduced
    ],
    'Testing Accuracy': [
        accuracy_score(y_test, y_pred_test),
        accuracy_test_rfe,
        accuracy_test_reduced
    ]
}).melt(id_vars='Model', var_name='Type', value_name='Accuracy')

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', hue='Type', data=results_df)
plt.title('Comparison of Decision Tree Performance Before and After Feature Reduction')
plt.xlabel('Model Type')
plt.ylabel('Accuracy')
plt.legend(title='Accuracy Type')
plt.ylim(0.95, 1.01)  # Adjust y-limits to zoom in on differences if they are minor
plt.show()

# Output Poisonous/Edible mushrooms for class balance check
count = data['class'].value_counts()
sns.set_context('talk')
plt.figure(figsize=(8, 7))
sns.barplot(x=count.index, y=count.values, alpha=0.8, palette="bwr")
plt.ylabel('Count')
plt.xlabel('Class')
plt.title('Overall Number of Poisonous and Edible Mushrooms')
plt.show()

#DT Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), color='b')  # Display top 20 features
plt.title('Feature Importances in Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Plot the comparison of model performance before and after feature reduction
results_df = pd.DataFrame({
    'Model': ['Original', 'RFE', 'Reduced Correlation'],
    'Training Accuracy': [
        accuracy_score(y_train, dt_classifier.predict(X_train)),
        accuracy_train_rfe,
        accuracy_train_reduced
    ],
    'Testing Accuracy': [
        accuracy_score(y_test, y_pred_test),
        accuracy_test_rfe,
        accuracy_test_reduced
    ]
}).melt(id_vars='Model', var_name='Type', value_name='Accuracy')

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', hue='Type', data=results_df)
plt.title('Comparison of Decision Tree Performance Before and After Feature Reduction')
plt.xlabel('Model Type')
plt.ylabel('Accuracy')
plt.legend(title='Accuracy Type')
plt.ylim(0.95, 1.01)  # Adjust y-limits to zoom in on differences if they are minor
plt.show()

# Output Poisonous/Edible mushrooms for class balance check
count = data['class'].value_counts()
sns.set_context('talk')
plt.figure(figsize=(8, 7))
sns.barplot(x=count.index, y=count.values, alpha=0.8, palette="bwr")
plt.ylabel('Count')
plt.xlabel('Class')
plt.title('Overall Number of Poisonous and Edible Mushrooms')
plt.show()

# DT Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), color='b')  # Display top 20 features
plt.title('Top 20 Feature Importances in Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

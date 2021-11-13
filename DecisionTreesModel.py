# importing the libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from CustomFunctions import *

# setting the random seed
SEED = 12

# loading the iris dataset
iris = load_iris()

# loading the data
X, y, features = load_data(iris)

# splitting the data into training and testing sets
train_X, train_y, test_X, test_y = split_data(X, y, test_size=0.2)

# creating the decision tree model
treeModel = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=SEED, min_samples_leaf=5, ccp_alpha=0.034)

# fitting the model
treeModel.fit(train_X, train_y)

# visualizing the decision tree
graph_tree(treeModel, features)

# predicting the test set
preds = treeModel.predict(test_X)

# printing the classification report of the decision tree predictions
print(classification_report(test_y, preds))
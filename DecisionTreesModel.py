from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from CustomFunctions import *


SEED = 12

iris = load_iris()

X, y, features = load_data(iris)


train_X, train_y, test_X, test_y = split_data(X, y, test_size=0.2)


treeModel = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=SEED, min_samples_leaf=5, ccp_alpha=0.034)


treeModel.fit(train_X, train_y)


graph_tree(treeModel, features)


preds = treeModel.predict(test_X)


print(classification_report(test_y, preds))
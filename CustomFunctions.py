from os import name
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
from IPython.display import Image
from six import StringIO
import graphviz, pydotplus


def load_data(data):
    features = data.feature_names
    X = data.data
    y = data.target
    return X, y, features

def split_data(X, y, test_size=0.20):
    split = int(len(X)*test_size)
    train_X, val_X = X[split:], X[:split]
    train_y, val_y = y[split:], y[:split]
    return train_X, train_y, val_X, val_y

def graph_tree(model, features):
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,feature_names=features,filled=True,rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())

if name == '__main__':
    data = load_iris()
    X, y, features = load_data(data, data.feature_names)
    train_X, train_y, val_X, val_y = split_data(X, y)
    print(len(train_X), len(train_y), len(val_X), len(val_y))
    
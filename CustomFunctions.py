# importing the required packages
from os import name
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
from IPython.display import Image
from six import StringIO
import graphviz, pydotplus


# Function to load the data into X and y and get the features names
def load_data(data):
    """
    Loads the data into X and y and returns the features names
    params:data: the data to be loaded
    """
    features = data.feature_names
    X = data.data
    y = data.target
    return X, y, features

# Function to split the data into train and test data
def split_data(X, y, test_size=0.20):
    """
    Splits the data into train and test data

    params: X: the data to be split into train and test set data
            y: the target column to be split into target train and test set
            test_size: the size of the test set
    """
    split = int(len(X)*test_size)
    train_X, val_X = X[split:], X[:split]
    train_y, val_y = y[split:], y[:split]
    return train_X, train_y, val_X, val_y

# Function to visualize the decision tree model
def graph_tree(model, features):
    """
    Visualizes the decision tree model

    params: model: the model to be visualized
            features: the features names to be used in the graph
    """
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,feature_names=features,filled=True,rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())


    
#     mndata = MNIST('supervised_learning/data/fashion')
#     mndata.gz = True


#     images: List
#     labels: array
#     images, labels = mndata.load_training()
# from mnist import MNIST
from typing import List
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt


def main():
    # Loading IRIS dataset from scikit-learn object into iris variable.
    iris = datasets.load_iris()

    # Prints the type/type object of iris
    print(type(iris))
    # <class 'sklearn.datasets.base.Bunch'>

    # prints the dictionary keys of iris data
    print(iris.keys())

    # prints the type/type object of given attributes
    print(type(iris.data), type(iris.target))

    # prints the no of rows and columns in the dataset
    print(iris.data.shape)

    # prints the target set of the data
    print(iris.target_names)

    # Load iris training dataset
    X = iris.data

    # Load iris target set
    Y = iris.target

    # Convert datasets' type into dataframe
    df = pd.DataFrame(X, columns=iris.feature_names)

    # Print the first five tuples of dataframe.
    print(df.head())

    pass

    

if __name__ == "__main__":
    main()

